"""Smoke tests — run with `pytest tests/`.

These tests verify wiring (state shapes, routing, parsing) WITHOUT calling a
real LLM. The LLM-touching pieces are exercised by `test_e2e.py` which is
gated on the OPENAI_API_KEY env var.
"""
from __future__ import annotations

import os

import pytest

from evaluation.evaluator import _parse_json
from evaluation.metrics import heuristic_score
from graph.state import initial_state
from graph.workflow import _route_after_eval
from utils import load_config


# ─────────────────────────── state ───────────────────────────
def test_initial_state_defaults():
    s = initial_state("test query")
    assert s["query"] == "test query"
    assert s["iteration"] == 0
    assert s["scores"] == {}
    assert s["max_iterations"] == 2


# ─────────────────────────── routing ─────────────────────────
@pytest.mark.parametrize(
    "score, iteration, max_iter, threshold, expected",
    [
        (9.0, 0, 2, 8.0, "pass"),     # above threshold → done
        (5.0, 0, 2, 8.0, "revise"),   # below + budget left → revise
        (5.0, 2, 2, 8.0, "pass"),     # budget exhausted → give up
        (8.0, 1, 2, 8.0, "pass"),     # exactly at threshold → done
    ],
)
def test_routing(score, iteration, max_iter, threshold, expected):
    state = {
        "weighted_score": score,
        "iteration": iteration,
        "max_iterations": max_iter,
        "quality_threshold": threshold,
    }
    assert _route_after_eval(state) == expected


# ─────────────────────────── evaluator parser ────────────────
def test_parse_json_clean():
    assert _parse_json('{"coherence": 8.5}') == {"coherence": 8.5}


def test_parse_json_with_fence():
    raw = "```json\n{\"clarity\": 7}\n```"
    assert _parse_json(raw) == {"clarity": 7}


def test_parse_json_with_prose():
    raw = "Here is the eval: {\"completeness\": 6.0} — done."
    assert _parse_json(raw) == {"completeness": 6.0}


def test_parse_json_garbage_returns_none():
    assert _parse_json("not json at all") is None


# ─────────────────────────── heuristics ──────────────────────
def test_heuristic_handles_empty():
    h = heuristic_score("")
    assert h["length_ok"] == 0.0
    assert h["structure_ok"] == 0.0


def test_heuristic_full_structure():
    body = "word " * 700
    fake = (
        "# Title\n## Introduction\n" + body
        + "\n## Insights\n## Analysis\n## Conclusion\n"
    )
    h = heuristic_score(fake)
    assert h["structure_ok"] == 10.0
    assert h["length_ok"] == 10.0


# ─────────────────────────── config ──────────────────────────
def test_config_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.llm.provider == "openai"
    assert cfg.workflow.max_iterations == 2


def test_per_agent_override():
    cfg = load_config("config.yaml")
    writer_cfg = cfg.llm_for("writer")
    # writer has temperature override in config.yaml
    assert writer_cfg.temperature == 0.5
    # base model preserved
    assert writer_cfg.model == cfg.llm.model


# ─────────────────────────── e2e (gated) ─────────────────────
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="No LLM API key configured",
)
def test_e2e_short_run():
    from utils import StepLogger
    from graph import run_workflow

    cfg = load_config("config.yaml")
    cfg.workflow.max_iterations = 0  # skip rewrite loop for speed
    logger = StepLogger(log_file=None)
    out = run_workflow(cfg, logger, query="What is the role of vector databases in RAG?")
    assert out["final_report"]
    assert out["weighted_score"] >= 0
