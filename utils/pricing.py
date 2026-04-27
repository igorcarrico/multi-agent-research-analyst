"""Token cost calculator.

Prices are in USD per 1 million tokens. Update as providers change pricing.
Source: official Anthropic and OpenAI pricing pages (2026).
"""
from __future__ import annotations

# (input_per_1M, output_per_1M) in USD
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5":          (1.00, 5.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-7":           (15.00, 75.00),
    # OpenAI
    "gpt-4o-mini":               (0.15, 0.60),
    "gpt-4o":                    (2.50, 10.00),
    "gpt-4.1-mini":              (0.40, 1.60),
}


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD. Returns 0 for unknown models (no false estimates)."""
    if model not in PRICES:
        return 0.0
    p_in, p_out = PRICES[model]
    return (input_tokens / 1_000_000) * p_in + (output_tokens / 1_000_000) * p_out


def summarize_run(events: list[dict]) -> dict:
    """Aggregate token usage and cost from a list of LogEvent dicts.

    Returns:
      {
        "by_agent": {agent_name: {input, output, cost_usd}},
        "totals": {input, output, cost_usd},
      }
    """
    by_agent: dict[str, dict] = {}
    total_in = total_out = 0
    total_cost = 0.0

    for ev in events:
        usage = (ev.get("payload") or {}).get("usage")
        if not usage:
            continue
        agent = ev.get("agent", "unknown")
        model = usage.get("model", "")
        in_t = int(usage.get("input_tokens", 0) or 0)
        out_t = int(usage.get("output_tokens", 0) or 0)
        cost = cost_for(model, in_t, out_t)

        slot = by_agent.setdefault(agent, {"input": 0, "output": 0, "cost_usd": 0.0})
        slot["input"] += in_t
        slot["output"] += out_t
        slot["cost_usd"] += cost

        total_in += in_t
        total_out += out_t
        total_cost += cost

    return {
        "by_agent": by_agent,
        "totals": {
            "input": total_in,
            "output": total_out,
            "cost_usd": round(total_cost, 6),
        },
    }
