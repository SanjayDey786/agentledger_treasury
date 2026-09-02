"""
agentledger.core.reallocator
==============================
Module B: Fluid Credit Reallocation / Yield Engine.

Implements the "Give Back" pattern (unused allocation deltas flow back to
the Treasury Pool automatically on release) and the Micro-Lending protocol
(an agent nearing its local ceiling but demonstrably close to finishing its
task can request a credit-line extension funded by yields harvested from
other, already-settled nodes).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from agentledger.core.treasury import AllocationToken, Treasury
from agentledger.exceptions import (
    CreditExtensionDeniedError,
    UnknownAllocationTokenError,
)


class ProgressHook:
    """
    Tracks an agent's task-completion progress (0.0 - 1.0) so the Reallocator
    can decide whether a credit extension is economically justified. In a
    real LangGraph/CrewAI integration this is fed by step counters, plan
    checklists, or token-count heuristics from the middleware wrapper.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._progress: float = 0.0

    def update(self, progress: float) -> None:
        self._progress = max(0.0, min(1.0, progress))

    @property
    def progress(self) -> float:
        return self._progress


class Reallocator:
    """
    Sits alongside a `Treasury` and provides the higher-level negotiation
    logic for micro-lending. It never mutates ledger state directly --
    all mutations are delegated to `Treasury.grant_extension` /
    `Treasury.release_allocation` so the treasury remains the single source
    of truth and audit trail.
    """

    #: An extension is only considered "worth funding" if the requesting
    #: agent has reported at least this much progress on its task. This
    #: encodes the PRD's "90% through a task" heuristic as a configurable
    #: default rather than a hardcoded magic number.
    DEFAULT_MIN_PROGRESS_FOR_EXTENSION: float = 0.75

    def __init__(
        self,
        treasury: Treasury,
        min_progress_for_extension: float = DEFAULT_MIN_PROGRESS_FOR_EXTENSION,
    ) -> None:
        self.treasury = treasury
        self.min_progress_for_extension = min_progress_for_extension
        self._progress_hooks: dict[str, ProgressHook] = {}

    # ------------------------------------------------------------------ #
    # Progress tracking
    # ------------------------------------------------------------------ #

    def progress_hook(self, agent_id: str) -> ProgressHook:
        """Get-or-create the `ProgressHook` for a given agent."""
        hook = self._progress_hooks.get(agent_id)
        if hook is None:
            hook = ProgressHook(agent_id)
            self._progress_hooks[agent_id] = hook
        return hook

    def report_progress(self, agent_id: str, progress: float) -> None:
        """
        Called by integration middleware (e.g. after each LangGraph node
        step) to update how far along an agent is in its task, expressed as
        a fraction in [0.0, 1.0].
        """
        self.progress_hook(agent_id).update(progress)

    # ------------------------------------------------------------------ #
    # Give-Back pattern
    # ------------------------------------------------------------------ #

    async def give_back(self, token: AllocationToken, actual_usd_spent: Decimal | float | str) -> Decimal:
        """
        Thin, semantically-named wrapper around `Treasury.release_allocation`
        for the "an agent finished and is yielding unused credit" case.
        Returns the amount yielded back to the pool.
        """
        return await self.treasury.release_allocation(token, actual_usd_spent)

    # ------------------------------------------------------------------ #
    # Micro-lending
    # ------------------------------------------------------------------ #

    async def request_credit_extension(
        self,
        agent_id: str,
        token: AllocationToken | str,
        required_usd: Decimal | float | str,
        *,
        force: bool = False,
    ) -> AllocationToken:
        """
        Request additional headroom on an already-active `AllocationToken`.

        The extension is granted only if:
          1. The Treasury has enough uncommitted `available_usd` to cover it
             (this money can only come from yields already given back by
             other agents, or from budget never committed in the first
             place -- the Treasury's accounting makes no distinction, which
             is exactly the "fluid pool" behavior the PRD specifies), AND
          2. Either `force=True` is passed by the caller (e.g. a supervisor
             override), or the agent's reported progress meets
             `min_progress_for_extension` -- preventing runaway agents that
             are barely started from draining the shared pool.

        Raises `CreditExtensionDeniedError` if either condition fails.
        """
        token_id = token.token_id if isinstance(token, AllocationToken) else token
        extra = Decimal(str(required_usd))

        if not force:
            hook = self._progress_hooks.get(agent_id)
            progress = hook.progress if hook is not None else 0.0
            if progress < self.min_progress_for_extension:
                reason = (
                    f"reported progress {progress:.0%} is below the "
                    f"{self.min_progress_for_extension:.0%} threshold required "
                    f"to justify a micro-lending extension"
                )
                self.treasury.record_extension_denial(agent_id, token_id, extra, reason)
                raise CreditExtensionDeniedError(agent_id, extra, reason=reason)

        available = self.treasury.available_usd
        if extra > available:
            reason = (
                f"only ${available:.6f} is currently uncommitted in the treasury pool, "
                f"which is less than the requested ${extra:.6f}"
            )
            self.treasury.record_extension_denial(agent_id, token_id, extra, reason)
            raise CreditExtensionDeniedError(agent_id, extra, reason=reason)

        try:
            return await self.treasury.grant_extension(token_id, extra)
        except UnknownAllocationTokenError:
            raise
