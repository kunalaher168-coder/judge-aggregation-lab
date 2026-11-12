"""Model tiers and list prices used for per-call cost accounting.

Prices are USD per million tokens, Anthropic first-party API list prices as
of 2026-11-03 (Haiku price is for prompts up to 100K tokens). Check the
current pricing page before relying on absolute dollar figures; relative
comparisons between tiers are what the routing analysis uses.
"""

from __future__ import annotations

TIERS: dict[str, str] = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# model id -> (input, output, cache_read) USD per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    # fallback targets that may serve a request when server-side fallbacks fire
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
}

CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute cache writes bill at 1.25x input


def model_for(tier_or_model: str) -> str:
    return TIERS.get(tier_or_model, tier_or_model)


def tier_for(model: str) -> str:
    for tier, mid in TIERS.items():
        if mid == model:
            return tier
    return model


def call_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD cost of one call. Output tokens include billed thinking tokens."""
    if model not in PRICES:
        raise KeyError(f"no price for model '{model}'; add it to judgelab.agents.pricing.PRICES")
    p_in, p_out, p_cache = PRICES[model]
    return (
        input_tokens * p_in
        + output_tokens * p_out
        + cache_read_tokens * p_cache
        + cache_write_tokens * p_in * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000
