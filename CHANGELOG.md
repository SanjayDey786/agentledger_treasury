# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] – 2026-09-02

### Added
- Core Treasury with in‑memory ledger (`agentledger_treasury.core.treasury`)
- Fluid credit reallocation and micro‑lending (`agentledger_treasury.core.reallocator`)
- ROI‑driven degradation engine (`agentledger_treasury.core.degrader`)
- Pricing tables and usage extraction helpers
- LangGraph node guard (`@ledger_guard`) and LangChain Runnable wrapper
- CrewAI task wrapper
- PostgreSQL backend for distributed, multi-worker deployments (`agentledger_treasury.backends.postgres.PostgresTreasury`), sharing a ledger across processes via row-level locking
- Full test suite and type checking
- Documentation and examples

### Changed
- Renamed the distribution and top-level package from `agentledger` to `agentledger_treasury` (PyPI name: `agentledger-treasury`)