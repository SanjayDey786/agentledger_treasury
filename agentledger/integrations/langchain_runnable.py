"""
agentledger.integrations.langchain_runnable
==============================================
Module D (LangChain half): wraps any object exposing LangChain's
`Runnable` protocol (`.invoke(input, config=None)` / `.ainvoke(...)`) with
the same pre-flight / degrade / post-flight / micro-lend behavior as the
LangGraph node guard, without importing `langchain_core` -- this module
duck-types the Runnable interface so it works against real LangChain
Runnables, CrewAI's internal LLM wrappers, or simple hand-rolled stand-ins
used in tests.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from agentledger.core.degrader import DegradationEngine
from agentledger.core.reallocator import Reallocator
from agentledger.core.treasury import AllocationToken, Treasury
from agentledger.exceptions import AllocationExceededError, CreditExtensionDeniedError
from agentledger.pricing import DEFAULT_PRICING_TABLE, PricingTable
from agentledger.usage import estimate_preflight_cost, extract_usage, settle_cost


class LedgerGuardedRunnable:
    """
    Wraps a LangChain-compatible `Runnable` (or any object with a
    `.ainvoke(input, config=None)` / `.invoke(input, config=None)` method)
    so every invocation is metered through the `Treasury`.

    The wrapped runnable's `config` dict is used to carry the resolved
    (possibly degraded) model name under `config["configurable"]["model"]`,
    matching LangChain's `RunnableConfig.configurable` convention, so
    downstream chat-model runnables that read their model from config pick
    up degradation transparently.
    """

    def __init__(
        self,
        runnable: Any,
        agent_id: str,
        treasury: Treasury,
        *,
        max_usd_ceiling: Decimal | float | str,
        default_model: str = "claude-3-5-sonnet",
        reallocator: Optional[Reallocator] = None,
        degradation_engine: Optional[DegradationEngine] = None,
        pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
        expected_output_tokens: int = 512,
        auto_request_extension: bool = True,
        extension_multiplier: Decimal | float = Decimal("0.5"),
    ) -> None:
        self.runnable = runnable
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

    def _resolve_model(self, config: Optional[Dict[str, Any]]) -> str:
        requested = self.default_model
        if config:
            requested = config.get("configurable", {}).get("model", requested)
        if self.degradation_engine is not None:
            return self.degradation_engine.resolve_model(self.agent_id, requested)
        return requested

    def _inject_model(self, config: Optional[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        new_config = dict(config) if config else {}
        configurable = dict(new_config.get("configurable", {}))
        configurable["model"] = model_name
        new_config["configurable"] = configurable
        return new_config

    async def _acquire(self, prompt_text: str, model_name: str) -> AllocationToken:
        preflight_cost = estimate_preflight_cost(
            prompt_text=prompt_text,
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

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        model_name = self._resolve_model(config)
        config = self._inject_model(config, model_name)
        prompt_text = input if isinstance(input, str) else str(input)

        token = await self._acquire(prompt_text, model_name)
        try:
            if hasattr(self.runnable, "ainvoke"):
                result = await self.runnable.ainvoke(input, config=config, **kwargs)
            else:
                result = self.runnable.invoke(input, config=config, **kwargs)
        except BaseException:
            await self.treasury.release_allocation(token, Decimal("0"))
            raise

        return await self._settle(token, result, model_name)

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Synchronous convenience entrypoint. Internally runs the async path
        to completion -- do not call this from inside a running event loop
        (use `ainvoke` there instead).
        """
        import asyncio

        return asyncio.run(self.ainvoke(input, config=config, **kwargs))


def guard_runnable(
    runnable: Any,
    agent_id: str,
    treasury: Treasury,
    *,
    max_usd_ceiling: Decimal | float | str,
    default_model: str = "claude-3-5-sonnet",
    reallocator: Optional[Reallocator] = None,
    degradation_engine: Optional[DegradationEngine] = None,
    pricing_table: PricingTable = DEFAULT_PRICING_TABLE,
    expected_output_tokens: int = 512,
) -> LedgerGuardedRunnable:
    """Functional convenience wrapper around `LedgerGuardedRunnable(...)`."""
    return LedgerGuardedRunnable(
        runnable,
        agent_id,
        treasury,
        max_usd_ceiling=max_usd_ceiling,
        default_model=default_model,
        reallocator=reallocator,
        degradation_engine=degradation_engine,
        pricing_table=pricing_table,
        expected_output_tokens=expected_output_tokens,
    )
