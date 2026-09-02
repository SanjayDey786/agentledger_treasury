"""
agentledger.usage
===================
Shared helpers for extracting LiteLLM-style token usage metadata from the
heterogeneous return shapes produced by LangGraph nodes, LangChain
Runnables, and CrewAI Task outputs, and turning that into a settled USD
cost via the `PricingTable`.

Every integration wrapper in `agentledger.integrations.*` funnels through
`extract_usage` / `settle_cost` so pricing logic lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from agentledger.pricing import DEFAULT_PRICING_TABLE, PricingTable


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model_name: Optional[str] = None


def extract_usage(result: Any, fallback_model: Optional[str] = None) -> Optional[TokenUsage]:
    """
    Best-effort extraction of token usage from a node/runnable/task result.

    Supports, in priority order:
      1. LiteLLM-style `response.usage` objects with `.prompt_tokens` /
         `.completion_tokens` attributes (what LangChain's `AIMessage.
         response_metadata["token_usage"]` and most raw provider SDKs use).
      2. A plain dict result containing a `"usage"` key shaped like
         `{"prompt_tokens": int, "completion_tokens": int, "model": str}`
         or the OpenAI/Anthropic-native `{"input_tokens", "output_tokens"}`.
      3. A LangChain `AIMessage`-like object exposing `.response_metadata`
         or `.usage_metadata`.

    Returns `None` if no usage information could be located -- callers
    should treat that as "unable to settle a precise cost" and fall back to
    a conservative estimate or a zero-cost settlement, per their own policy.
    """
    usage_dict = None
    model_name = fallback_model

    # Case 1 & 3: LangChain AIMessage-like objects.
    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage_dict = response_metadata.get("token_usage") or response_metadata.get("usage")
        model_name = response_metadata.get("model_name", model_name)

    usage_metadata = getattr(result, "usage_metadata", None)
    if usage_dict is None and isinstance(usage_metadata, dict):
        usage_dict = usage_metadata

    # Case 2: plain dict-shaped state/results.
    if usage_dict is None and isinstance(result, dict):
        usage_dict = result.get("usage")
        model_name = result.get("model", model_name)

    # Direct `.usage` attribute (raw SDK response objects).
    if usage_dict is None:
        raw_usage = getattr(result, "usage", None)
        if raw_usage is not None:
            usage_dict = raw_usage if isinstance(raw_usage, dict) else raw_usage.__dict__

    if usage_dict is None:
        return None

    input_tokens = (
        usage_dict.get("prompt_tokens")
        if usage_dict.get("prompt_tokens") is not None
        else usage_dict.get("input_tokens", 0)
    )
    output_tokens = (
        usage_dict.get("completion_tokens")
        if usage_dict.get("completion_tokens") is not None
        else usage_dict.get("output_tokens", 0)
    )
    model_name = usage_dict.get("model", model_name)

    return TokenUsage(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        model_name=model_name,
    )


def settle_cost(
    usage: Optional[TokenUsage],
    fallback_model: str,
    pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
) -> Decimal:
    """
    Convert a `TokenUsage` into a settled USD amount using `pricing_table`.
    If `usage` is `None` (no metadata could be extracted), returns
    `Decimal("0")` -- the caller's allocation ceiling is yielded back in
    full rather than silently overcharging the session for an unmeasurable
    call.
    """
    if usage is None:
        return Decimal("0")
    model = usage.model_name or fallback_model
    return pricing_table.estimate_cost(model, usage.input_tokens, usage.output_tokens)


def estimate_preflight_cost(
    prompt_text: str,
    model_name: str,
    expected_output_tokens: int = 512,
    pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
    chars_per_token: float = 4.0,
) -> Decimal:
    """
    Pre-flight cost estimate used before an LLM call has actually happened
    (so no real usage metadata exists yet). Uses a conservative
    characters-per-token heuristic for the input side and a caller-supplied
    expected-output-token budget, matching the common LiteLLM pattern of
    estimating cost before dispatch to decide whether to even attempt a
    call under the current allocation ceiling.
    """
    estimated_input_tokens = max(1, int(len(prompt_text) / chars_per_token))
    return pricing_table.estimate_cost(model_name, estimated_input_tokens, expected_output_tokens)
