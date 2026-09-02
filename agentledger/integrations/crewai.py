"""
agentledger.integrations.crewai
==================================
Module D (CrewAI half): wraps a CrewAI `Task`'s callable execution unit
(commonly exposed as `Task.execute_sync` / `Task.execute_async`, or simply
the plain callback function you pass into a manually-driven Crew) with the
Treasury/Reallocator/DegradationEngine pipeline.

CrewAI agents are typically driven from synchronous, thread-pool-based
execution rather than asyncio, so `LedgerGuardedCrewTask` exposes a
synchronous `run(...)` entrypoint that manages its own event loop via
`asyncio.run`, in addition to an `arun(...)` coroutine for callers already
inside an event loop (e.g. a CrewAI flow driven through `crewai.flow`,
which is asyncio-based).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Callable, Optional

from agentledger.core.degrader import DegradationEngine
from agentledger.core.reallocator import Reallocator
from agentledger.core.treasury import AllocationToken, Treasury
from agentledger.exceptions import AllocationExceededError, CreditExtensionDeniedError
from agentledger.pricing import DEFAULT_PRICING_TABLE, PricingTable
from agentledger.usage import estimate_preflight_cost, extract_usage, settle_cost

TaskFn = Callable[..., Any]


class LedgerGuardedCrewTask:
    """
    Wraps a callable representing a single CrewAI task execution
    (signature: `(agent, task_description: str, **kwargs) -> result`, or
    any compatible callable -- the wrapper only inspects `task_description`
    for pre-flight cost estimation and the return value for post-flight
    usage extraction).
    """

    def __init__(
        self,
        task_fn: TaskFn,
        agent_id: str,
        treasury: Treasury,
        *,
        max_usd_ceiling: Decimal | float | str,
        default_model: str = "claude-3-5-sonnet",
        reallocator: Optional[Reallocator] = None,
        degradation_engine: Optional[DegradationEngine] = None,
        pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
        expected_output_tokens: int = 768,
        auto_request_extension: bool = True,
        extension_multiplier: Decimal | float = Decimal("0.5"),
    ) -> None:
        self.task_fn = task_fn
        self.agent_id = agent_id
        self.treasury = treasury
        self.max_usd_ceiling = Decimal(str(max_usd_ceiling))
        self.default_model = default_model
        self.reallocator = reallocator
        self.degradation_engine = degradation_engine
        self.pricing_table = pricing_table
        self.expected_output_tokens = expected_output_tokens
        self.auto_request_extension = auto_request_extension
        self.extension_multiplier = Decimal(str(extension_multiplier))

    def _resolve_model(self, requested_model: Optional[str]) -> str:
        model = requested_model or self.default_model
        if self.degradation_engine is not None:
            return self.degradation_engine.resolve_model(self.agent_id, model)
        return model

    async def _acquire(self, task_description: str, model_name: str) -> AllocationToken:
        preflight_cost = estimate_preflight_cost(
            prompt_text=task_description,
            model_name=model_name,
            expected_output_tokens=self.expected_output_tokens,
            pricing_table=self.pricing_table,
        )
        ceiling = max(self.max_usd_ceiling, preflight_cost)
        return await self.treasury.request_allocation(self.agent_id, ceiling, model_name=model_name)

    async def _settle(self, token: AllocationToken, result: Any, model_name: str) -> Any:
        usage = extract_usage(result, fallback_model=model_name)
        actual_cost = settle_cost(usage, fallback_model=model_name, pricing_table=self.pricing_table)

        if actual_cost > token.ceiling_usd and self.auto_request_extension and self.reallocator is not None:
            shortfall = actual_cost - token.ceiling_usd
            extension = shortfall + (shortfall * self.extension_multiplier)
            try:
                await self.reallocator.request_credit_extension(self.agent_id, token, extension, force=True)
            except CreditExtensionDeniedError:
                await self.treasury.release_allocation(token, token.ceiling_usd)
                raise AllocationExceededError(self.agent_id, token.token_id, actual_cost, token.ceiling_usd)

        await self.treasury.release_allocation(token, actual_cost)
        return result

    async def arun(
        self,
        agent: Any,
        task_description: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        model_name = self._resolve_model(model)
        token = await self._acquire(task_description, model_name)
        try:
            result = self.task_fn(agent, task_description, model=model_name, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        except BaseException:
            await self.treasury.release_allocation(token, Decimal("0"))
            raise

        return await self._settle(token, result, model_name)

    def run(self, agent: Any, task_description: str, *, model: Optional[str] = None, **kwargs: Any) -> Any:
        """Synchronous entrypoint for CrewAI's default thread-pool executor."""
        return asyncio.run(self.arun(agent, task_description, model=model, **kwargs))


def guard_crew_task(
    task_fn: TaskFn,
    agent_id: str,
    treasury: Treasury,
    *,
    max_usd_ceiling: Decimal | float | str,
    default_model: str = "claude-3-5-sonnet",
    reallocator: Optional[Reallocator] = None,
    degradation_engine: Optional[DegradationEngine] = None,
    pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
    expected_output_tokens: int = 768,
) -> LedgerGuardedCrewTask:
    """Functional convenience wrapper around `LedgerGuardedCrewTask(...)`."""
    return LedgerGuardedCrewTask(
        task_fn,
        agent_id,
        treasury,
        max_usd_ceiling=max_usd_ceiling,
        default_model=default_model,
        reallocator=reallocator,
        degradation_engine=degradation_engine,
        pricing_table=pricing_table,
        expected_output_tokens=expected_output_tokens,
    )
