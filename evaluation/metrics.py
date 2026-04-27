"""Cheap heuristic signals. Not a substitute for LLM-as-judge, but a useful
sanity check that catches obvious failure modes (empty output, missing sections).
"""
from __future__ import annotations

import re

REQUIRED_SECTIONS = ("Introduction", "Insights", "Analysis", "Conclusion")


def heuristic_score(report: str) -> dict[str, float]:
    """Returns a dict of cheap-to-compute signals on a 0–10 scale."""
    if not report:
        return {"length_ok": 0.0, "structure_ok": 0.0, "lexical_diversity": 0.0}

    words = re.findall(r"\w+", report)
    n = len(words)
    # 600–900 words is the target range from the writer prompt.
    if 600 <= n <= 900:
        length_ok = 10.0
    elif 400 <= n < 600 or 900 < n <= 1200:
        length_ok = 6.0
    else:
        length_ok = 3.0

    found = sum(1 for s in REQUIRED_SECTIONS if s.lower() in report.lower())
    structure_ok = 10.0 * found / len(REQUIRED_SECTIONS)

    # Type-token ratio — repetitive writing scores low.
    unique = len({w.lower() for w in words})
    ttr = unique / n if n else 0
    lexical_diversity = min(10.0, ttr * 20)  # ttr ~0.5 → 10

    return {
        "length_ok": round(length_ok, 2),
        "structure_ok": round(structure_ok, 2),
        "lexical_diversity": round(lexical_diversity, 2),
    }
