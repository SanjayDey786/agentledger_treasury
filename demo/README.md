# AgentLedger Treasury -- Postgres demo

Three scripts, run against a **real** local Postgres (not mocked), that
answer two questions honestly: where does this project actually help, and
where does it not.

## Setup

```bash
docker run -d --name agentledger-demo-pg \
  -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=agentledger_demo \
  -p 55432:5432 postgres:16-alpine

pip install -e ".[postgres]"   # from the repo root
```

All three scripts read `DEMO_POSTGRES_DSN` (default:
`postgresql://demo:demo@localhost:55432/agentledger_demo`).

LLM calls are simulated ([`fake_llm.py`](fake_llm.py)) rather than real API
calls -- deterministic, free, and it lets each scenario control exactly how
much output-token variance to inject, which is what actually makes the
"useful vs. not useful" comparison legible.

## Scenario A -- `python scenario_a_distributed.py`

What a distributed backend buys you that the in-memory `Treasury` cannot,
even in principle:

- **A1**: 4 *separate OS processes* draw against a nominal $1.00 budget.
  With each process holding its own private in-memory `Treasury`, they
  collectively grant $3.60 -- 3.6x over budget, because nothing coordinates
  them. Pointed at one shared `PostgresTreasury` session instead, real
  row-level locking holds the total at or under $1.00, no matter how many
  processes race.
- **A2**: a process allocates money, then exits without settling
  (simulating a crash). A brand-new process, with zero memory of the first,
  opens the same `session_id` and sees the exact correct remaining budget.
  State lives in Postgres, not in a process's RAM.

## Scenario B -- `python scenario_b_rescue.py` (the useful case)

150 trials, each replaying the *same* random per-agent token-usage draws
through two governance policies: a naive fixed even split of the budget,
vs. the shared treasury with give-back + micro-lending enabled. Typical
result: **naive succeeds ~28% of trials, the ledger policy succeeds ~59%**
-- roughly double the session success rate, using the same total budget,
because unused slack from agents that finished early gets redirected to
agents that ran over, instead of sitting stranded.

## Scenario C -- `python scenario_c_limits.py` (the honest limits)

- **C1**: same setup as Scenario B, but with zero nominal slack and much
  higher variance, so essentially no agent reliably finishes under budget.
  Both policies collapse toward the same low success rate (~7% vs ~19% in
  a typical run, compared to Scenario B's 28% vs 59%) -- proving
  reallocation redistributes existing slack, it does not conjure money the
  session was never given.
- **C2**: a single agent with no siblings hits a simulated 8x runaway
  completion. `request_credit_extension` is correctly denied -- there is no
  other agent's give-back anywhere in the pool to fund the shortfall. This
  also makes concrete a point from the design discussion: by the time the
  ledger denies the extension, the (simulated) call has already happened;
  in a real deployment the provider would have already billed for those
  tokens. The treasury can't claw that back -- it can only decide how your
  application reacts next.

## What this demo incidentally found

Running the Postgres backend against a real database (rather than the
mocked pool used during development) surfaced three real bugs that are now
fixed in `agentledger_treasury/backends/postgres.py`:

1. Ledger-entry timestamps were passed as raw Python floats to a
   `TIMESTAMPTZ` column (`asyncpg` requires a `datetime`).
2. Ledger-entry metadata dicts were passed unserialized to a `JSONB`
   column (`asyncpg` requires a JSON string, both encoding and decoding).
3. `CREATE TABLE IF NOT EXISTS`, run from several processes racing to
   initialize the schema for the first time, could raise a duplicate-key
   error against Postgres's own catalog -- fixed with a `pg_advisory_lock`
   around schema creation, which serializes it across processes, not just
   within one.

None of these were caught by the unit tests or the mocked-pool smoke test
from the original build, because neither exercises real `asyncpg` type
encoding or real multi-process contention. That's worth remembering as its
own lesson: a distributed-systems feature isn't verified until it's been
run distributed, against the real dependency.
