"""Streamlit UI.

Run with: streamlit run ui/app.py

Layout:
  - Sidebar: config (model, threshold, iterations, comparison toggle)
  - Top: query input + run button
  - Body (after run):
      * Side-by-side: Baseline | Multi-agent
      * Score table + verdict
      * Execution trace (logs)
      * Final report (full width)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make the project root importable when launched via `streamlit run ui/app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

from agents import BaselineAgent
from evaluation import compare_outputs
from graph import run_workflow
from utils import StepLogger, load_config, summarize_run
from utils.pdf import markdown_to_pdf

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Analyst",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Multi-Agent Research Analyst")
st.caption(
    "LangChain + LangGraph · 5 collaborating agents · LLM-as-judge evaluation · "
    "side-by-side baseline comparison"
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — config
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")
    cfg = load_config(ROOT / "config.yaml")

    language = st.selectbox(
        "Idioma da resposta",
        ["Português (Brasil)", "English", "Español", "Français"],
        index=0,
        help=(
            "Define em qual idioma os agentes vão escrever pesquisa, análise, "
            "relatório e veredicto. Não afeta o idioma da pergunta — pode "
            "perguntar em inglês e receber resposta em português."
        ),
    )

    cfg.llm.model = st.text_input(
        "Modelo LLM",
        cfg.llm.model,
        help=(
            "Nome do modelo usado por todos os agentes. "
            "Ex: claude-haiku-4-5-20251001 (rápido/barato), "
            "claude-sonnet-4-6 (equilíbrio), claude-opus-4-7 (máxima qualidade)."
        ),
    )

    cfg.llm.temperature = st.slider(
        "Temperatura (criatividade)", 0.0, 1.0, cfg.llm.temperature, 0.1,
        help=(
            "Controla o quão 'criativo' o modelo é. "
            "**0.0** = sempre a mesma resposta, mais previsível, melhor pra tarefas factuais. "
            "**1.0** = respostas mais variadas e criativas, melhor pra brainstorm. "
            "**Recomendado: 0.3** pra pesquisa séria."
        ),
    )

    cfg.workflow.max_iterations = st.slider(
        "Máximo de loops de revisão", 0, 5, cfg.workflow.max_iterations,
        help=(
            "Quantas vezes o Reviewer pode reescrever o relatório se ele não "
            "atingir a nota mínima. **0** = sem revisão (mais rápido/barato). "
            "**2-3** = bom equilíbrio. Cada loop adiciona ~15s e ~$0.01."
        ),
    )

    cfg.workflow.quality_threshold = st.slider(
        "Nota mínima de qualidade (0–10)", 0.0, 10.0, cfg.workflow.quality_threshold, 0.5,
        help=(
            "Nota que o relatório precisa atingir pra ser considerado 'pronto'. "
            "Se ficar abaixo, dispara um loop de revisão (até o limite acima). "
            "**8.0** = exigente mas atingível. **9.0+** = quase sempre vai forçar revisão."
        ),
    )

    cfg.workflow.enable_web_search = st.checkbox(
        "Buscar na web (DuckDuckGo)", value=cfg.workflow.enable_web_search,
        help=(
            "Se ativado, o Research Agent busca snippets reais na web antes "
            "de gerar as notas. Se desativado, usa só o conhecimento do modelo. "
            "Ative pra tópicos recentes; desative pra tópicos clássicos."
        ),
    )

    run_comparison = st.checkbox(
        "Comparar com baseline (single-agent)", value=True,
        help=(
            "Se ativado, roda em paralelo um segundo pipeline simples (1 prompt "
            "só) e compara os dois lado a lado com scores. Demonstra por que o "
            "multi-agent vale a pena. Dobra o custo da execução."
        ),
    )

    st.divider()
    st.caption(
        f"**Provider:** `{cfg.llm.provider}` · chaves no `.env` · "
        "ajustes por agente em `config.yaml`"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Tópico de pesquisa",
    placeholder="ex: Impacto da IA nas profissões em 2026",
    help="Pode escrever na sua língua. A resposta vai sair no idioma escolhido na sidebar.",
)
context = st.text_area(
    "Contexto opcional (restrições, público-alvo, etc.)",
    height=80,
    help="Opcional. Use pra refinar — ex: 'foco em RH', 'público técnico', 'horizonte 5 anos'.",
)

run = st.button("▶️ Rodar análise", type="primary", disabled=not query.strip())


# Map UI labels to canonical names the LLM understands well.
_LANG_MAP = {
    "Português (Brasil)": "Brazilian Portuguese",
    "English": "English",
    "Español": "Spanish",
    "Français": "French",
}


def _run_baseline(cfg, query, language):
    logger = StepLogger(log_file=ROOT / "logs/baseline.jsonl")
    agent = BaselineAgent(cfg, logger)
    out = agent.run({"query": query, "language": language})
    return out["report"], logger


def _run_multi(cfg, query, context, language):
    logger = StepLogger(log_file=ROOT / "logs/multiagent.jsonl")
    final = run_workflow(cfg, logger, query=query, context=context, language=language)
    return final, logger


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────
if run and query.strip():
    t_start = time.time()
    lang_canonical = _LANG_MAP[language]

    with st.status("Rodando pipelines…", expanded=True) as status:
        if run_comparison:
            st.write("Disparando baseline + multi-agent em paralelo…")
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_base = ex.submit(_run_baseline, cfg, query, lang_canonical)
                f_multi = ex.submit(_run_multi, cfg, query, context, lang_canonical)
                baseline_report, baseline_logger = f_base.result()
                multi_state, multi_logger = f_multi.result()
            st.write("Ambos terminaram. Avaliando…")
            comp_logger = StepLogger(log_file=ROOT / "logs/comparison.jsonl")
            comparison = compare_outputs(
                cfg,
                comp_logger,
                query=query,
                analysis=multi_state.get("analysis", ""),
                baseline_report=baseline_report,
                multiagent_report=multi_state.get("final_report", ""),
                language=lang_canonical,
            )
        else:
            st.write("Rodando pipeline multi-agent…")
            multi_state, multi_logger = _run_multi(cfg, query, context, lang_canonical)
            baseline_report, baseline_logger, comparison = None, None, None

        status.update(
            label=f"Concluído em {time.time() - t_start:.1f}s", state="complete", expanded=False
        )

    # ─────────────────────────── Comparison panel ──────────────────────────
    if comparison:
        st.subheader("⚖️ Comparação lado a lado")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### :grey[Baseline (1 prompt só)]")
            st.metric(
                "Nota de qualidade (0–10)",
                comparison["baseline"]["weighted_score"],
            )
            with st.expander("Ver relatório do baseline", expanded=False):
                st.markdown(baseline_report)
        with col_b:
            st.markdown("### :green[Multi-Agent (5 agentes)]")
            st.metric(
                "Nota de qualidade (0–10)",
                comparison["multiagent"]["weighted_score"],
                delta=f"{comparison['delta']:+.2f} vs baseline",
            )
            with st.expander("Ver relatório multi-agent", expanded=False):
                st.markdown(multi_state["final_report"])

        st.markdown("#### Notas por eixo")
        st.dataframe(
            {
                "eixo": list(comparison["baseline"]["scores"].keys()),
                "baseline": list(comparison["baseline"]["scores"].values()),
                "multi-agent": list(comparison["multiagent"]["scores"].values()),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 👨‍⚖️ Veredicto do juiz")
        st.info(comparison["verdict"])

    # ─────────────────────────── Cost & token panel ────────────────────────
    st.subheader("💰 Custo e tokens")
    all_events = (
        [e.to_dict() for e in multi_logger.events]
        + ([e.to_dict() for e in baseline_logger.events] if baseline_logger else [])
        + ([e.to_dict() for e in comp_logger.events] if comparison else [])
    )
    summary = summarize_run(all_events)
    totals = summary["totals"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens entrada", f"{totals['input']:,}")
    c2.metric("Tokens saída", f"{totals['output']:,}")
    c3.metric("Custo total (USD)", f"${totals['cost_usd']:.4f}")
    if summary["by_agent"]:
        st.markdown("**Por agente:**")
        st.dataframe(
            [
                {
                    "agente": a,
                    "input": v["input"],
                    "output": v["output"],
                    "custo (USD)": round(v["cost_usd"], 5),
                }
                for a, v in summary["by_agent"].items()
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ─────────────────────────── Execution trace ───────────────────────────
    st.subheader("🔍 Trace de execução (multi-agent)")
    with st.expander("Ver passo a passo dos agentes", expanded=False):
        for ev in multi_logger.events:
            icon = {"start": ":arrow_forward:", "end": ":white_check_mark:",
                    "decision": ":bulb:", "error": ":x:"}.get(ev.event, ":small_blue_diamond:")
            dur = f" ({ev.duration_ms:.0f} ms)" if ev.duration_ms else ""
            st.markdown(
                f"{icon} **{ev.agent}** · `{ev.event}`{dur} — "
                f"{ev.payload.get('summary', '')}"
            )
            if ev.event == "decision":
                st.json(ev.payload)

    # ─────────────────────────── Final report ──────────────────────────────
    st.subheader("📄 Relatório final")
    st.markdown(multi_state["final_report"])

    if multi_state.get("scores"):
        st.caption(
            f"Nota final: **{multi_state['weighted_score']}/10** · "
            f"iterações de revisão: {multi_state.get('iteration', 0)} · "
            f"por eixo: {multi_state['scores']}"
        )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Baixar Markdown",
            data=multi_state["final_report"],
            file_name="relatorio.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl2:
        try:
            pdf_bytes = markdown_to_pdf(multi_state["final_report"], title=query)
            st.download_button(
                "📄 Baixar PDF",
                data=pdf_bytes,
                file_name="relatorio.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Falha ao gerar PDF: {e}")

else:
    st.info("👆 Digite um tópico acima e clique em **Rodar análise**.")
