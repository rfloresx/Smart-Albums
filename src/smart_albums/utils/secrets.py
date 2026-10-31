"""Read secrets from the environment with a Docker-friendly ``_FILE`` convention.

A plain env var (``FOO``) exposes its value in ``docker inspect`` and the
container's environment; a Docker/Kubernetes *secret* is instead mounted as a
file whose path is passed in ``FOO_FILE``. :func:`read_env_secret` prefers the
``_FILE`` variant and falls back to the plain var, matching the convention
PhotoStudio already uses for ``IMMICH_API_KEY_FILE``.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_env_secret(name: str, default: str = "") -> str:
    """Return the value of secret ``name``, preferring a ``<name>_FILE`` path.

    Resolution order:

    1. ``<name>_FILE`` — if set and non-empty, its value is a filesystem path;
       the file's contents (stripped of surrounding whitespace/newline) are
       returned. A missing or unreadable file raises ``OSError`` so a
       misconfigured secret fails loudly rather than silently falling back.
    2. ``<name>`` — the plain env var.
    3. ``default``.

    Args:
        name: The base env var name (e.g. ``"IMMICH_API_KEY"``).
        default: Value to return when neither variant is set.

    Returns:
        The resolved secret value, or ``default`` if unset.

    Raises:
        OSError: If ``<name>_FILE`` is set but the file can't be read.
    """
    file_path = os.environ.get(f"{name}_FILE")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return os.environ.get(name, default)
