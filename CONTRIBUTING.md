# Contributing

Thanks for your interest in contributing to smart-albums!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rfloresx/Smart-Albums.git
cd Smart-Albums

# Create virtual environment
python3 -m venv .venv

# Install in editable mode with dev extras
.venv/bin/python -m pip install -e ".[dev]"

# Verify the setup
.venv/bin/pytest --co -q
```

## Running Tests

```bash
# All tests
.venv/bin/pytest

# Verbose output
.venv/bin/pytest -v

# Single file
.venv/bin/pytest tests/test_partition_time.py

# Type checking
.venv/bin/mypy src/smart_albums/
```

## Code Style

- **Type hints**: Annotate all function signatures. Use `from __future__ import annotations`.
- **Docstrings**: Module-level docstrings on every file. Google-style for classes/methods.
- **Logging**: Use `logger = logging.getLogger(__name__)`. Never print from library code.
- **Enums**: Prefer `str` enums (`class Foo(str, enum.Enum)`) for serializable values.
- **Imports**: Guard heavy imports behind `if TYPE_CHECKING:` when used only for annotations.

## Project Structure

- `src/smart_albums/` — Main package (src layout)
- `src/smart_albums/core/` — Framework: runner, registry, context, cache, protocols
- `src/smart_albums/nodes/` — Pipeline stages organized by category
- `src/smart_albums/clients/` — External service clients (Immich, Ollama, HuggingFace)
- `src/smart_albums/pipelines/` — Declarative pipeline definitions
- `src/webgui/` — NiceGUI web interface
- `tests/` — Unit and property-based tests

## Adding a New Stage

1. Create a file in the appropriate `nodes/` category package.
2. Subclass `Stage` and decorate with `@stage("category.name")`.
3. Define `_config_schema` as a tuple of `ConfigParam` instances.
4. Implement the `async def run(self, ctx: PipelineContext) -> ContextBatch:` method.
5. Register the import in `src/smart_albums/cli/imports.py`.
6. Add a test file `tests/test_<module>.py`.

## Adding a New Client

1. Create a file in `src/smart_albums/clients/`.
2. Implement the relevant protocol (`IImageClient`, `ILLMClient`, `IEmbeddingClient`).
3. Decorate with `@ProtocolsRegistry.register("name", IProtocol)`.
4. Add the import to `src/smart_albums/clients/__init__.py`.

## Testing Conventions

- One test file per source module.
- Use factory helpers from `conftest.py` (`make_asset`, `make_context`, etc.).
- Group related tests into classes.
- Use `@pytest.mark.asyncio` for async tests (auto mode enabled).
- No mocking frameworks — inject test doubles via constructor.

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes and ensure all tests pass.
3. Run `mypy src/smart_albums/` and fix any type errors.
4. Write a clear commit message describing what changed and why.
5. Open a pull request against `main`.

## Reporting Issues

Open an issue on GitHub with:
- What you were trying to do
- What happened instead
- Steps to reproduce
- Your environment (Python version, OS, Docker or local)
