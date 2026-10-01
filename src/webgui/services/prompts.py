"""Prompt management — file-based CRUD for scoring prompt files.

Prompt files are plain Markdown files stored in ``state.config.prompts_dir``.
They are referenced by the pipeline via the ``analyze.score`` stage's
``prompt_file`` config parameter (see ``smart_albums.nodes.analyze.score``),
which expects a filesystem path to a ``.md`` file containing the prompt text
sent to the vision LLM.

This service only manages the files themselves (list/create/update/delete).
Filenames are restricted to a safe character set to prevent path traversal
outside ``prompts_dir``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from webgui.state import state

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-z0-9_-]+")


class PromptError(ValueError):
    """Raised for invalid prompt names or content."""


@dataclass(frozen=True)
class PromptInfo:
    """A single prompt file.

    Attributes:
        name: The filename (including ``.md`` extension) — the stable
            identifier used to reference this prompt from pipeline config.
        label: A human-friendly title derived from the filename.
        content: The full Markdown text of the prompt.
        updated_at: Last-modified time as a Unix timestamp.
    """

    name: str
    label: str
    content: str
    updated_at: float


def _prompts_dir() -> Path:
    d = Path(state.config.prompts_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _label_for(name: str) -> str:
    """Derive a display label from a filename (strip extension, title-case)."""
    return Path(name).stem.replace("_", " ").replace("-", " ").strip().title()


def normalize_name(raw_name: str) -> str:
    """Normalize user input into a safe ``.md`` filename.

    Lowercases, replaces whitespace with underscores, strips any character
    outside ``[a-z0-9_-]``, and ensures a ``.md`` extension. This also
    prevents path traversal since path separators are stripped.

    Args:
        raw_name: The name as entered by the user (with or without extension).

    Returns:
        A safe filename, e.g. ``"My Prompt!"`` -> ``"my_prompt.md"``.

    Raises:
        PromptError: If the resulting name is empty.
    """
    stem = raw_name.strip()
    if stem.lower().endswith(".md"):
        stem = stem[: -len(".md")]
    stem = stem.strip().lower().replace(" ", "_")
    stem = _SAFE_NAME_RE.sub("", stem)
    if not stem:
        raise PromptError("Prompt name must contain at least one letter, digit, '_' or '-'.")
    return f"{stem}.md"


def _safe_path(name: str) -> Path:
    """Resolve a prompt filename to a path guaranteed to stay in prompts_dir.

    Args:
        name: A filename previously produced by :func:`normalize_name` (or
            returned by :func:`list_prompts`).

    Returns:
        The absolute path to the prompt file.

    Raises:
        PromptError: If the name escapes the prompts directory.
    """
    base = _prompts_dir().resolve()
    candidate = (base / name).resolve()
    if candidate.parent != base:
        raise PromptError(f"Invalid prompt name: {name!r}")
    return candidate


def resolve_prompt_path(name: str) -> Path:
    """Public wrapper around :func:`_safe_path` for other services to use.

    Resolves a prompt filename (as stored in ``pipeline_settings['score']
    ['prompt_file']``) to an absolute path confined to ``prompts_dir``.
    Used by ``services/jobs.py`` when building the CLI config for a job, so
    a crafted or malformed ``prompt_file`` value can't make the CLI read an
    arbitrary file on the host (see services/jobs.py's _build_cli_config).

    Raises:
        PromptError: If the name is empty, absolute, contains a path
            separator, or otherwise resolves outside ``prompts_dir``.
    """
    return _safe_path(name)


def list_prompts() -> list[PromptInfo]:
    """Return all prompt files, sorted by name.

    Returns:
        A list of :class:`PromptInfo`, one per ``*.md`` file in prompts_dir.
    """
    results: list[PromptInfo] = []
    for item in sorted(_prompts_dir().glob("*.md")):
        try:
            content = item.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read prompt file %s: %s", item, exc)
            continue
        results.append(
            PromptInfo(
                name=item.name,
                label=_label_for(item.name),
                content=content,
                updated_at=item.stat().st_mtime,
            )
        )
    return results


def get_prompt(name: str) -> PromptInfo | None:
    """Fetch a single prompt by filename.

    Args:
        name: The prompt filename (e.g. ``"my_prompt.md"``).

    Returns:
        The :class:`PromptInfo`, or ``None`` if it does not exist.
    """
    path = _safe_path(name)
    if not path.exists():
        return None
    return PromptInfo(
        name=path.name,
        label=_label_for(path.name),
        content=path.read_text(encoding="utf-8"),
        updated_at=path.stat().st_mtime,
    )


def create_prompt(raw_name: str, content: str) -> PromptInfo:
    """Create a new prompt file.

    Args:
        raw_name: User-entered name (normalized via :func:`normalize_name`).
        content: The prompt Markdown text.

    Returns:
        The created :class:`PromptInfo`.

    Raises:
        PromptError: If the name is invalid, content is empty, or a prompt
            with the same normalized name already exists.
    """
    if not content.strip():
        raise PromptError("Prompt content cannot be empty.")

    name = normalize_name(raw_name)
    path = _safe_path(name)
    if path.exists():
        raise PromptError(f"A prompt named {name!r} already exists.")

    path.write_text(content, encoding="utf-8")
    logger.info("Created prompt %s", name)
    return get_prompt(name)  # type: ignore[return-value]


def update_prompt(name: str, content: str) -> PromptInfo:
    """Overwrite an existing prompt's content.

    Args:
        name: The prompt filename to update.
        content: The new Markdown text.

    Returns:
        The updated :class:`PromptInfo`.

    Raises:
        PromptError: If content is empty or the prompt does not exist.
    """
    if not content.strip():
        raise PromptError("Prompt content cannot be empty.")

    path = _safe_path(name)
    if not path.exists():
        raise PromptError(f"Prompt {name!r} not found.")

    path.write_text(content, encoding="utf-8")
    logger.info("Updated prompt %s", name)
    return get_prompt(name)  # type: ignore[return-value]


def delete_prompt(name: str) -> None:
    """Delete a prompt file.

    Args:
        name: The prompt filename to delete.

    Raises:
        PromptError: If the prompt does not exist.
    """
    path = _safe_path(name)
    if not path.exists():
        raise PromptError(f"Prompt {name!r} not found.")
    path.unlink()
    logger.info("Deleted prompt %s", name)
