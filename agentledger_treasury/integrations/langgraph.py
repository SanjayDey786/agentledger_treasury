"""
agentledger_treasury.integrations.langgraph
=====================================
Module D (LangGraph half): a decorator / middleware class that wraps any
LangGraph node function (sync or async, `state -> state` or
`state -> Awaitable[state]`) so that it:

  1. Pre-flight: estimates the cost of the upcoming call and requests an
     `AllocationToken` from the `Treasury` before the node ever runs.
  2. Applies ROI-driven degradation (Module C) to the outgoing model
     parameter if the treasury is in a critical burn state.
  3. Executes the wrapped node.
  4. Post-flight: extracts real token usage from the node's return value,
     settles the actual spend against the ledger, and yields back any
     unused delta (Module B's "Give Back" pattern).
  5. Mid-run over-budget handling: if the node itself raises because it
     detected it would blow past its ceiling (or if post-flight settlement
     reveals the actual spend exceeded the granted ceiling), the guard first
     attempts a micro-lending credit extension via `Reallocator` before
     propagating the failure.

This module has zero hard dependency on `langgraph` or `langchain-core`
being installed -- it operates purely on plain dicts / arbitrary state
objects and duck-types LangGraph's `state -> state` node contract, so it
works whether or not those packages are present in the environment.
"""

from __future__ import annotations

import functools
import inspect
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from agentledger_treasury.core.reallocator import Reallocator
from agentledger_treasury.core.treasury import AllocationToken, Treasury
from agentledger_treasury.core.degrader import DegradationEngine
from agentledger_treasury.exceptions import AllocationExceededError, CreditExtensionDeniedError
from agentledger_treasury.pricing import DEFAULT_PRICING_TABLE, PricingTable
from agentledger_treasury.usage import estimate_preflight_cost, extract_usage, settle_cost

# LangGraph node state is, in practice, always a `dict`-like mapping (or any
# object the graph's reducer functions know how to merge), so we type it as
# `Any` rather than introducing a `TypeVar` -- a generic here would need to
# be bound per-decorated-function, which `Callable`-returning decorator
# factories can't express cleanly, and would provide no real type safety
# over `Any` in exchange for the added complexity.
NodeFn = Callable[..., Union[Any, Awaitable[Any]]]

# Key used to look up the prompt text and requested model on the incoming
# state dict for pre-flight cost estimation, when the caller doesn't supply
# explicit `prompt_key` / `model_key` overrides.
_DEFAULT_PROMPT_KEY = "input"
_DEFAULT_MODEL_KEY = "model"


class LedgerGuardMiddleware:
    """
    Stateful wrapper object bound to one `agent_id` and one `Treasury`
    (plus optional `Reallocator` / `DegradationEngine`). Prefer the
    `@ledger_guard(...)` decorator factory below for typical use; construct
    this directly if you need to wrap a node dynamically at runtime (e.g.
    LangGraph nodes assembled programmatically from a config file).
    """

    def __init__(
        self,
        agent_id: str,
        treasury: Treasury,
        *,
        max_usd_ceiling: Decimal | float | str,
        reallocator: Optional[Reallocator] = None,
        degradation_engine: Optional[DegradationEngine] = None,
        pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
        prompt_key: str = _DEFAULT_PROMPT_KEY,
        model_key: str = _DEFAULT_MODEL_KEY,
        expected_output_tokens: int = 512,
        auto_request_extension: bool = True,
        extension_multiplier: Decimal | float = Decimal("0.5"),
    ) -> None:
        self.agent_id = agent_id
        self.treasury = treasury
        self.max_usd_ceiling = Decimal(str(max_usd_ceiling))
        self.reallocator = reallocator
        self.degradation_engine = degradation_engine
        self.pricing_table = pricing_table
        self.prompt_key = prompt_key
        self.model_key = model_key
        self.expected_output_tokens = expected_output_tokens
        self.auto_request_extension = auto_request_extension
        self.extension_multiplier = Decimal(str(extension_multiplier))

    def _resolve_model(self, state: Dict[str, Any]) -> str:
        requested_model = state.get(self.model_key) if isinstance(state, dict) else None
        requested_model = requested_model or "claude-3-5-sonnet"
        if self.degradation_engine is not None:
            return self.degradation_engine.resolve_model(self.agent_id, requested_model)
        return requested_model

    def _preflight_estimate(self, state: Dict[str, Any], model_name: str) -> Decimal:
        prompt_text = state.get(self.prompt_key, "") if isinstance(state, dict) else str(state)
        return estimate_preflight_cost(
            prompt_text=str(prompt_text),
            model_name=model_name,
            expected_output_tokens=self.expected_output_tokens,
            pricing_table=self.pricing_table,
        )

    async def _acquire_token(self, state: Dict[str, Any], model_name: str) -> AllocationToken:
        preflight_cost = self._preflight_estimate(state, model_name)
        ceiling = max(self.max_usd_ceiling, preflight_cost)
        return await self.treasury.request_allocation(self.agent_id, ceiling, model_name=model_name)

    async def _settle(self, token: AllocationToken, result: Any, model_name: str) -> Any:
        usage = extract_usage(result, fallback_model=model_name)
        actual_cost = settle_cost(usage, fallback_model=model_name, pricing_table=self.pricing_table)

        if actual_cost > token.ceiling_usd and self.auto_request_extension and self.reallocator is not None:
            shortfall = actual_cost - token.ceiling_usd
            extension = shortfall + (shortfall * self.extension_multiplier)
            try:
                await self.reallocator.request_credit_extension(
                    self.agent_id, token, extension, force=True
                )
            except CreditExtensionDeniedError:
                await self.treasury.release_allocation(token, token.ceiling_usd)
                raise AllocationExceededError(
                    self.agent_id, token.token_id, actual_cost, token.ceiling_usd
                )

        await self.treasury.release_allocation(token, actual_cost)
        return result

    async def _handle_failure(self, token: AllocationToken, error: BaseException) -> None:
        """
        On a mid-run failure, still settle the token so its already-consumed
        headroom (if any progress was tracked) is accounted for, then yield
        back whatever remains. We conservatively settle at the token's
        current ceiling only if the error itself is a budget error;
        otherwise we settle at 0 spend (the call failed before consuming
        billable tokens, e.g. a network error).
        """
        if isinstance(error, (AllocationExceededError, CreditExtensionDeniedError)):
            await self.treasury.release_allocation(token, token.ceiling_usd)
        else:
            await self.treasury.release_allocation(token, Decimal("0"))

    def wrap(self, fn: NodeFn) -> Callable[[Any], Awaitable[Any]]:
        is_coro = inspect.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
            state_dict = state if isinstance(state, dict) else {}
            model_name = self._resolve_model(state_dict)

            # Apply the (possibly degraded) model back onto the state so the
            # wrapped node actually uses the cheaper tier when instructed.
            if isinstance(state, dict) and self.model_key in state:
                state = {**state, self.model_key: model_name}

            token = await self._acquire_token(state_dict, model_name)

            async def _run() -> Any:
                return await fn(state, *args, **kwargs) if is_coro else fn(state, *args, **kwargs)

            try:
                result = await _run()
            except AllocationExceededError as exc:
                # Mid-run over-budget: attempt one micro-lending rescue, then
                # retry the node exactly once before giving up.
                if not (self.auto_request_extension and self.reallocator is not None):
                    await self._handle_failure(token, exc)
                    raise
                try:
                    await self.reallocator.request_credit_extension(
                        self.agent_id, token, exc.attempted_usd - exc.ceiling_usd, force=True
                    )
                except CreditExtensionDeniedError:
                    await self._handle_failure(token, exc)
                    raise
                result = await _run()
            except BaseException as exc:
                await self._handle_failure(token, exc)
                raise

            return await self._settle(token, result, model_name)

        return wrapped

    def __call__(self, fn: NodeFn) -> Callable[[Any], Awaitable[Any]]:
        return self.wrap(fn)


def ledger_guard(
    agent_id: str,
    treasury: Treasury,
    *,
    max_usd_ceiling: Decimal | float | str,
    reallocator: Optional[Reallocator] = None,
    degradation_engine: Optional[DegradationEngine] = None,
    pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
    prompt_key: str = _DEFAULT_PROMPT_KEY,
    model_key: str = _DEFAULT_MODEL_KEY,
    expected_output_tokens: int = 512,
    auto_request_extension: bool = True,
    extension_multiplier: Decimal | float = Decimal("0.5"),
) -> Callable[[NodeFn], Callable[[Any], Awaitable[Any]]]:
    """
    Decorator factory for wrapping a LangGraph node function (or any plain
    `state -> state` / `state -> Awaitable[state]` callable) with full
    AgentLedger pre-flight allocation, ROI-driven degradation, post-flight
    settlement, and micro-lending fallback.

    Example
    -------
    >>> treasury = Treasury(session_id="sess-1", total_session_budget_usd="1.00")
    >>> reallocator = Reallocator(treasury)
    >>>
    >>> @ledger_guard("researcher", treasury, max_usd_ceiling="0.20", reallocator=reallocator)
    ... async def researcher_node(state: dict) -> dict:
    ...     ...  # call the LLM, return updated state including `usage`
    """
    middleware = LedgerGuardMiddleware(
        agent_id,
        treasury,
        max_usd_ceiling=max_usd_ceiling,
        reallocator=reallocator,
        degradation_engine=degradation_engine,
        pricing_table=pricing_table,
        prompt_key=prompt_key,
        model_key=model_key,
        expected_output_tokens=expected_output_tokens,
        auto_request_extension=auto_request_extension,
        extension_multiplier=extension_multiplier,
    )

    def decorator(fn: NodeFn) -> Callable[[Any], Awaitable[Any]]:
        return middleware.wrap(fn)

    return decorator
