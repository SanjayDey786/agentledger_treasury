"""
agentledger.pricing
====================
Standardized pricing tables used to compute pre-flight and post-flight USD
costs for LLM calls, and to define the "cheaper tier" swap targets used by
the ROI-Driven Degradation Engine.

Costs are expressed in USD per single token (not per 1K/1M) as `Decimal` for
exact arithmetic -- avoiding floating point drift across thousands of
micro-transactions in a long agentic session.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional

from agentledger.exceptions import UnknownModelPricingError


def _per_million(usd_per_million: float) -> Decimal:
    """Convert a human-friendly '$ per 1M tokens' figure into per-token Decimal cost."""
    return Decimal(str(usd_per_million)) / Decimal("1000000")


@dataclass(frozen=True)
class ModelPrice:
    """Per-token pricing plus degradation metadata for a single model."""

    model_name: str
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    # The model to swap to when the Degradation Engine forces a cheaper tier.
    # `None` means this model is already the cheapest in its family.
    degrade_to: Optional[str] = None
    # Relative capability tier, lower is cheaper/weaker. Used to prevent
    # accidentally "degrading" to a more expensive model due to bad config.
    tier_rank: int = 0

    def cost_of(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            self.input_cost_per_token * Decimal(input_tokens)
            + self.output_cost_per_token * Decimal(output_tokens)
        )


class PricingTable:
    """
    Thread-safe registry mapping model name -> ModelPrice.

    Ships with a default LiteLLM-style table for common Anthropic and OpenAI
    model families, including their degrade-to targets for Module C.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._table: Dict[str, ModelPrice] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        defaults = [
            ModelPrice(
                model_name="claude-opus-4",
                input_cost_per_token=_per_million(15.00),
                output_cost_per_token=_per_million(75.00),
                degrade_to="claude-3-5-sonnet",
                tier_rank=3,
            ),
            ModelPrice(
                model_name="claude-3-5-sonnet",
                input_cost_per_token=_per_million(3.00),
                output_cost_per_token=_per_million(15.00),
                degrade_to="claude-3-5-haiku",
                tier_rank=2,
            ),
            ModelPrice(
                model_name="claude-3-5-haiku",
                input_cost_per_token=_per_million(0.80),
                output_cost_per_token=_per_million(4.00),
                degrade_to=None,
                tier_rank=1,
            ),
            ModelPrice(
                model_name="gpt-4o",
                input_cost_per_token=_per_million(2.50),
                output_cost_per_token=_per_million(10.00),
                degrade_to="gpt-4o-mini",
                tier_rank=2,
            ),
            ModelPrice(
                model_name="gpt-4o-mini",
                input_cost_per_token=_per_million(0.15),
                output_cost_per_token=_per_million(0.60),
                degrade_to=None,
                tier_rank=1,
            ),
        ]
        for price in defaults:
            self._table[price.model_name] = price

    def register(self, price: ModelPrice) -> None:
        """Add or override a model's pricing entry. Thread-safe."""
        with self._lock:
            self._table[price.model_name] = price

    def get(self, model_name: str) -> ModelPrice:
        with self._lock:
            price = self._table.get(model_name)
        if price is None:
            raise UnknownModelPricingError(model_name)
        return price

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Decimal:
        return self.get(model_name).cost_of(input_tokens, output_tokens)

    def degrade_target(self, model_name: str) -> Optional[str]:
        """Return the next-cheaper model name, or None if already at the floor tier."""
        return self.get(model_name).degrade_to

    def all_models(self) -> Dict[str, ModelPrice]:
        with self._lock:
            return dict(self._table)


# Module-level singleton used by default across the framework. Callers that
# need isolated pricing (e.g. multi-tenant deployments) should instantiate
# their own `PricingTable()` and pass it explicitly to `Treasury(...)`.
DEFAULT_PRICING_TABLE = PricingTable()
