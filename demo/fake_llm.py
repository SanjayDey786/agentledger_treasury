"""
A deterministic, seedable stand-in for a real LLM call.

Real output-token counts are inherently unpredictable before generation
finishes -- that unpredictability is the entire reason a pre-flight cost
estimate can never be exact (see `agentledger_treasury.usage.
estimate_preflight_cost`). Rather than pretend a real API call would be
more "honest," this simulates that same unpredictability with a controlled
random distribution, so every scenario in this demo can be re-run
deterministically (same `seed` => same numbers => reproducible before/after
comparisons) while still exhibiting realistic variance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class LLMCallResult:
    prompt_tokens: int
    completion_tokens: int
    model: str


def call_llm(
    rng: random.Random,
    *,
    model: str,
    prompt_tokens: int,
    expected_output_tokens: int,
    variance: float = 0.35,
    runaway_probability: float = 0.0,
    runaway_multiplier: float = 8.0,
) -> LLMCallResult:
    """
    Simulate one LLM call's real token usage.

    `expected_output_tokens` is what a caller *thought* it would need
    (the same number they'd hand to `estimate_preflight_cost`). The real
    output is drawn from a log-normal distribution centered on that
    expectation with `variance` controlling spread -- real completions are
    rarely exactly what you guessed, and are more likely to overshoot than
    undershoot by a lot (fat right tail), which is why log-normal rather
    than normal is used here.

    `runaway_probability`: chance this call ignores the expectation
    entirely and produces `runaway_multiplier`x the expected tokens --
    modeling the "model got into a loop / verbose edge case" scenario that
    no pre-flight estimate can see coming.
    """
    if rng.random() < runaway_probability:
        completion_tokens = max(1, int(expected_output_tokens * runaway_multiplier))
    else:
        # log-normal centered so the median equals expected_output_tokens
        mu = 0.0
        sigma = variance
        multiplier = rng.lognormvariate(mu, sigma)
        completion_tokens = max(1, int(expected_output_tokens * multiplier))

    return LLMCallResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def as_usage_dict(result: LLMCallResult) -> dict:
    """Shape matching what `agentledger_treasury.usage.extract_usage` expects."""
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "model": result.model,
    }
