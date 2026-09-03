# AgentLedger Treasury

**A financial clearinghouse and dynamic credit-allocation middleware for stateful, hierarchical multi-agent workflows.**

AgentLedger Treasury treats a multi-agent session's token budget not as a static ceiling handed down to each node, but as a fluid corporate treasury: agents draw allocations against it, unspent capital yields back automatically, agents that are close to finishing but running tight can borrow against the pool's collective savings, and when the whole session's burn rate outpaces its actual progress, AgentLedger Treasury forces a downshift to cheaper models rather than letting the session blow its budget or stall out.

Built for [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) Runnables, and [CrewAI](https://github.com/joaomdmoura/crewAI) -- but the core (`agentledger_treasury.core`) has zero dependency on any of them, so it works with any orchestration framework that can call an `async` function around a sub-task.

## Why

Static per-agent budgets are wasteful in either direction: set them too tight and agents crash mid-task the moment a plan needs an extra reasoning pass; set them too loose and a runaway sub-agent can burn the entire session's budget before a supervisor ever notices. AgentLedger Treasury's answer is a shared, mutable pool with rules:

- **Give Back** -- an agent that finishes under its allocated ceiling immediately returns the unused delta to the shared pool.
- **Micro-Lending** -- an agent that is verifiably close to finishing (tracked via a progress hook) but hits its local ceiling can request a credit-line extension, funded only by capital other agents have already yielded back or that was never committed.
- **ROI-Driven Degradation** -- if the pool's remaining budget drops below a critical threshold *and* the session's spend is outpacing its actual step-by-step progress, AgentLedger Treasury transparently swaps subsequent LLM calls to a cheaper model tier instead of letting the session fail outright.

## Install

```bash
pip install -e .                  # core only
pip install -e ".[langchain]"     # + langchain-core / langgraph
pip install -e ".[crewai]"        # + crewai
pip install -e ".[postgres]"      # + asyncpg, for the distributed PostgreSQL backend
pip install -e ".[dev]"           # + pytest, mypy, ruff
```

## Architecture

| Module | Responsibility |
|---|---|
| `agentledger_treasury.core.treasury` | Module A -- the thread-safe, atomic, async-native central ledger. `Treasury.request_allocation` / `.release_allocation` / `.grant_extension`. |
| `agentledger_treasury.core.reallocator` | Module B -- the Give-Back pattern and micro-lending negotiation logic (`Reallocator.give_back`, `.request_credit_extension`, progress hooks). |
| `agentledger_treasury.core.degrader` | Module C -- burn-rate monitoring and forced model downgrades (`DegradationEngine.resolve_model`, `BudgetVelocityTracker`). |
| `agentledger_treasury.pricing` | LiteLLM-style per-token pricing table with configurable `degrade_to` chains (e.g. `claude-3-5-sonnet -> claude-3-5-haiku`). |
| `agentledger_treasury.usage` | Shared token-usage extraction / cost settlement helpers used by every integration wrapper. |
| `agentledger_treasury.integrations.langgraph` | Module D -- the `@ledger_guard` decorator for LangGraph nodes / plain `state -> state` callables. |
| `agentledger_treasury.integrations.langchain_runnable` | Module D -- `LedgerGuardedRunnable` wrapping any LangChain-compatible `Runnable`. |
| `agentledger_treasury.integrations.crewai` | Module D -- `LedgerGuardedCrewTask` wrapping a CrewAI task execution callable. |
| `agentledger_treasury.backends.postgres` | Module E -- `PostgresTreasury`, a drop-in, PostgreSQL-backed Treasury for multi-worker/multi-host deployments. |

## Distributed deployment with PostgreSQL

The in-memory `Treasury` lives inside a single process, so it can't be shared across multiple workers (pods, hosts, or processes) working the same session. `PostgresTreasury` stores all ledger state in PostgreSQL instead, using `SELECT ... FOR UPDATE` row locking so that concurrent `request_allocation` / `release_allocation` / `grant_extension` calls from *any* worker sharing the same `session_id` and connection pool are serialized and can never oversubscribe the pool.

```bash
pip install "agentledger_treasury[postgres]"
```

```python
import asyncpg
from agentledger_treasury import PostgresTreasury

pool = await asyncpg.create_pool(dsn="postgresql://...")
treasury = PostgresTreasury(
    pool=pool,
    session_id="distributed-session-1",
    total_session_budget_usd="10.00",  # required the first time a session is created
    create_tables=True,                # lazily creates the schema on first use (default)
)
```

`Reallocator` and `DegradationEngine` work unmodified against a `PostgresTreasury` -- they only touch the Treasury's public surface. A few methods necessarily differ from the in-memory `Treasury` because they now involve real I/O:

- `get_token`, `snapshot`, and `ledger_history` are `async def` on `PostgresTreasury` (they are synchronous on the in-memory `Treasury`, which never does I/O).
- `available_usd`, `committed_usd`, `settled_spent_usd`, `utilization_ratio`, and `remaining_ratio` stay **synchronous properties** (so `Reallocator`/`DegradationEngine` can keep calling them without `await`), backed by a small local cache that is refreshed on every allocation/release/extension this instance performs. Because other workers can mutate the same session between two of *this* instance's operations, treat them as eventually consistent -- call `await treasury.refresh()` first if a decision needs a guaranteed-fresh read.

The in-memory `Treasury` remains the recommended choice for single-process or demo use; reach for `PostgresTreasury` once you need multiple workers sharing one ledger.

## Quick start

```python
import asyncio
from agentledger_treasury import Treasury, Reallocator, DegradationEngine, BudgetVelocityTracker
from agentledger_treasury.integrations import ledger_guard

treasury = Treasury(session_id="sess-1", total_session_budget_usd="1.00")
reallocator = Reallocator(treasury)
velocity = BudgetVelocityTracker(expected_total_steps=5)
degrader = DegradationEngine(treasury, velocity, critical_remaining_threshold="0.20")

@ledger_guard("researcher", treasury, max_usd_ceiling="0.20",
              reallocator=reallocator, degradation_engine=degrader)
async def researcher_node(state: dict) -> dict:
    # ... call your LLM here, return state including a `usage` dict
    # (prompt_tokens / completion_tokens / model), LiteLLM-style.
    return {**state, "usage": {"prompt_tokens": 300, "completion_tokens": 200, "model": state["model"]}}

async def main():
    result = await researcher_node({"input": "...", "model": "claude-3-5-sonnet"})
    print(treasury.snapshot())

asyncio.run(main())
```

Run the full end-to-end rescue scenario (a supervisor with two workers, where one worker's savings bail out the other's overrun) with no external dependencies or API keys:

```bash
python examples/supervisor_worker_example.py
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Design notes

- **Decimal everywhere.** All USD amounts are `decimal.Decimal`, never `float`, to avoid drift across thousands of micro-transactions in a long-running session.
- **Concurrency.** Ledger bookkeeping is pure, in-memory, non-blocking arithmetic, so it is guarded by a single `threading.RLock` that is safe to acquire from both plain OS threads (CrewAI's default executor) and `async def` methods on an event loop -- while the public API remains fully `async/await`-native as required by highly parallel agent execution paths. See the docstring in `agentledger_treasury/core/treasury.py` for the full rationale.
- **Single source of truth.** `Treasury` is the only component that mutates ledger state. `Reallocator` and `DegradationEngine` are pure orchestration/policy layers on top of it, and every integration wrapper funnels cost extraction through `agentledger_treasury.usage` so pricing logic lives in exactly one place.
- **No silent overcharging.** If an integration wrapper cannot extract usage metadata from a call's result, the settled cost defaults to `$0`, and the full ceiling is yielded back -- AgentLedger Treasury never guesses at spend it can't verify.

## Contributing

Contributions are welcome -- see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, code style, and pull request process.

## License

MIT -- see [LICENSE](LICENSE).
