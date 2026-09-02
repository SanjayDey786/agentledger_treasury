"""
agentledger.core.degrader
============================
Module C: ROI-Driven Degradation Engine.

Monitors the velocity of the central Treasury Pool. If the session budget
drops below a configurable threshold *and* the workflow's step velocity is
outstripping budget expectations (i.e. burning cash faster than the
workflow is progressing toward completion), this engine intercepts routing
decisions and forces the orchestrator to swap the LLM model for subsequent
sub-tasks to a cheaper tier, using `agentledger.pricing.PricingTable` as the
source of degrade-to targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from agentledger.core.treasury import Treasury
from agentledger.pricing import DEFAULT_PRICING_TABLE, PricingTable


@dataclass(frozen=True)
class DegradationEvent:
    """Emitted every time the engine forces a model swap."""

    timestamp: float
    agent_id: str
    original_model: str
    degraded_model: str
    treasury_remaining_ratio: Decimal
    reason: str


class BudgetVelocityTracker:
    """
    Tracks the relationship between "budget burned" and "workflow steps
    completed" so the Degradation Engine can distinguish a session that is
    merely spending money from one that is burning money *faster than it is
    making progress* -- the actual trigger condition specified in the PRD
    ("workflow steps are outstripping budget expectations").
    """

    def __init__(self, expected_total_steps: int) -> None:
        if expected_total_steps <= 0:
            raise ValueError("expected_total_steps must be a positive integer")
        self.expected_total_steps = expected_total_steps
        self._completed_steps = 0

    def record_step_completed(self) -> None:
        self._completed_steps += 1

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def expected_progress_ratio(self) -> Decimal:
        """Fraction of total expected steps completed so far, in [0, 1]."""
        ratio = Decimal(self._completed_steps) / Decimal(self.expected_total_steps)
        return min(ratio, Decimal("1"))

    def is_burn_outstripping_progress(self, budget_utilization_ratio: Decimal) -> bool:
        """
        True when the fraction of budget already used exceeds the fraction
        of expected workflow steps completed -- i.e. the session is
        "ahead" on spend relative to how far along it actually is.
        """
        return budget_utilization_ratio > self.expected_progress_ratio


class DegradationEngine:
    """
    Wraps a `Treasury` and a `BudgetVelocityTracker` to decide, on each
    routing decision, whether a sub-task's model should be swapped down to a
    cheaper tier before execution.

    Usage pattern: integration middleware calls `resolve_model(agent_id,
    requested_model)` immediately before dispatching an LLM call, and uses
    the returned model name (which may equal the requested one if no
    degradation is warranted).
    """

    def __init__(
        self,
        treasury: Treasury,
        velocity_tracker: BudgetVelocityTracker,
        pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
        *,
        critical_remaining_threshold: Decimal | float = Decimal("0.20"),
        on_degrade: Optional[Callable[[DegradationEvent], None]] = None,
    ) -> None:
        """
        `critical_remaining_threshold`: once the treasury's remaining budget
        ratio drops at or below this value (default 20%, matching the PRD
        example), degradation becomes *eligible* to trigger. It still only
        actually fires if `velocity_tracker.is_burn_outstripping_progress`
        is also true for that moment -- pure low-budget-but-on-schedule
        sessions are left alone so mission completion isn't sacrificed
        prematurely.
        """
        self.treasury = treasury
        self.velocity_tracker = velocity_tracker
        self.pricing_table = pricing_table
        self.critical_remaining_threshold = Decimal(str(critical_remaining_threshold))
        self.on_degrade = on_degrade
        self._events: List[DegradationEvent] = []
        # Per-agent memory of the last model we forced them onto, so repeated
        # calls within the same critical window don't re-degrade an already
        # degraded model past its floor tier unnecessarily.
        self._forced_models: Dict[str, str] = {}

    @property
    def events(self) -> List[DegradationEvent]:
        return list(self._events)

    def _is_critical(self) -> bool:
        remaining = self.treasury.remaining_ratio
        utilization = self.treasury.utilization_ratio
        return (
            remaining <= self.critical_remaining_threshold
            and self.velocity_tracker.is_burn_outstripping_progress(utilization)
        )

    def resolve_model(self, agent_id: str, requested_model: str) -> str:
        """
        Returns the model name that should actually be used for this call.
        If the treasury is in a critical, burn-outstripping-progress state,
        this returns the pricing table's `degrade_to` target instead of
        `requested_model` (or `requested_model` unchanged if it is already
        at the cheapest tier, since there is nowhere further to degrade).
        """
        if not self._is_critical():
            return requested_model

        # If we already forced this agent onto a cheaper model earlier in
        # the critical window, keep using that one rather than re-resolving
        # from the original (which would otherwise just re-degrade the same
        # step repeatedly and emit duplicate events).
        current = self._forced_models.get(agent_id, requested_model)
        degrade_target = self.pricing_table.degrade_target(current)

        if degrade_target is None:
            # Already at the cheapest tier available for this model family.
            return current

        event = DegradationEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            original_model=requested_model,
            degraded_model=degrade_target,
            treasury_remaining_ratio=self.treasury.remaining_ratio,
            reason=(
                f"treasury remaining ratio {self.treasury.remaining_ratio:.2%} <= "
                f"threshold {self.critical_remaining_threshold:.2%} and budget "
                f"utilization is outstripping workflow progress"
            ),
        )
        self._events.append(event)
        self._forced_models[agent_id] = degrade_target
        if self.on_degrade is not None:
            self.on_degrade(event)
        return degrade_target

    def rewrite_runtime_config(self, agent_id: str, runtime_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience helper for integration wrappers: given an LLM runtime
        config dict (e.g. a LangChain `RunnableConfig`-style mapping or a
        raw kwargs dict containing a `"model"` key), returns a *new* dict
        with the `"model"` key rewritten if degradation is warranted. Does
        not mutate the input dict.
        """
        requested_model = runtime_config.get("model")
        if not requested_model:
            return dict(runtime_config)
        resolved = self.resolve_model(agent_id, requested_model)
        new_config = dict(runtime_config)
        new_config["model"] = resolved
        return new_config
