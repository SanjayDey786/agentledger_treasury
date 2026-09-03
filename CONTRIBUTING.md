# Contributing to AgentLedger

Thank you for your interest in improving AgentLedger! We welcome contributions of all kinds: bug reports, documentation, feature requests, and code.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment (Python 3.11+ recommended).
3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev,postgres]"
   ```
4. Install pre‑commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Code Style
- Formatting: Ruff (configured in pyproject.toml)

- Type annotations: mypy with strict mode

- Use Decimal for all monetary values – never floats.

## Testing

Run the test suite:
```bash
pytest tests/ -v
```
Ensure all tests pass and coverage does not decrease.

## Pull Request Process
1. Create a new branch for your feature or fix.

2. Write tests for any new functionality.

3. Update documentation and add a changelog entry under the Unreleased section.

4. Ensure CI passes (linting, type checking, tests across Python versions).

5. Open a pull request against the main branch, describing your changes and referencing any related issues.

## Code of Conduct
Please be respectful and constructive. We follow the Contributor Covenant.