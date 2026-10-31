"""Authentication module for the NiceGUI web interface.

Provides password hashing, session verification, and a guard function
for protecting pages. Uses NiceGUI's app.storage.user for session state.

IMPORTANT: app.storage.user is only available after the client WebSocket
connects. Always call `await ui.context.client.connected()` before
reading storage in a page handler.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from nicegui import app

# The bcrypt password primitives (hash/verify, off-event-loop dispatch, and
# the timing-safe unknown-username comparison) are shared across both web
# apps and live in protocols_system.webauth. They are re-exported here so
# existing callers keep importing them from webgui.auth; this module keeps
# only the Smart-Albums-specific login rate limiting and NiceGUI session
# helpers.
from protocols_system.webauth import (
    hash_password,
    hash_password_async,
    verify_password,
    verify_password_async,
)

logger = logging.getLogger(__name__)

__all__ = [
    "hash_password",
    "hash_password_async",
    "verify_password",
    "verify_password_async",
    "SESSION_VERSION_KEY",
    "check_login_rate_limit",
    "record_login_failure",
    "clear_login_failures",
    "is_authenticated",
    "get_current_username",
    "set_authenticated",
    "clear_authentication",
]

# Settings-table key that holds the current global session version. Bumped
# whenever a password changes so that other active sessions are logged out.
SESSION_VERSION_KEY = "session_version"


# ---------------------------------------------------------------------------
# Login rate limiting
# ---------------------------------------------------------------------------

# Per-identifier (username, lowercased) failure tracking for simple
# exponential backoff. In-memory only: resets on process restart, and (by
# design, since there's no per-IP tracking here) doesn't survive multiple
# app workers — adequate for this single-process NiceGUI deployment, not a
# substitute for a dedicated rate-limiting layer in front of it.
_login_failures: dict[str, list[float]] = {}
_MAX_TRACKED_FAILURES = 10
_FAILURE_WINDOW_SECONDS = 15 * 60


def _backoff_seconds(failure_count: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped at 30s."""
    return float(min(2 ** max(0, failure_count - 1), 30))


def check_login_rate_limit(identifier: str) -> Optional[float]:
    """Return seconds to wait before the next login attempt, or None if allowed.

    Call before attempting to verify credentials for ``identifier``
    (typically the submitted username, lowercased). Does not record
    anything by itself — call :func:`record_login_failure` on failure and
    :func:`clear_login_failures` on success.
    """
    now = time.monotonic()
    attempts = _login_failures.get(identifier)
    if not attempts:
        return None
    # Drop attempts outside the tracking window.
    attempts[:] = [t for t in attempts if now - t < _FAILURE_WINDOW_SECONDS]
    if not attempts:
        _login_failures.pop(identifier, None)
        return None
    wait = _backoff_seconds(len(attempts)) - (now - attempts[-1])
    return float(wait) if wait > 0 else None


def record_login_failure(identifier: str) -> None:
    """Record a failed login attempt for backoff purposes."""
    attempts = _login_failures.setdefault(identifier, [])
    attempts.append(time.monotonic())
    if len(attempts) > _MAX_TRACKED_FAILURES:
        del attempts[: len(attempts) - _MAX_TRACKED_FAILURES]


def clear_login_failures(identifier: str) -> None:
    """Clear failure tracking for ``identifier`` after a successful login."""
    _login_failures.pop(identifier, None)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def is_authenticated() -> bool:
    """Check if the current user session is authenticated.

    Also validates that the session_version (if one is stored in the DB)
    matches the session_version in the user's storage. This ensures that
    password changes invalidate other active sessions.
    """
    return bool(app.storage.user.get("authenticated", False))


def get_current_username() -> Optional[str]:
    """Return the username of the authenticated user, or None."""
    if is_authenticated():
        return app.storage.user.get("username")
    return None


async def set_authenticated(username: str) -> None:
    """Mark the current browser session as authenticated.

    Also stamps the session with the *current* global session version, so
    the freshly-authenticated session is not immediately treated as stale by
    ``AuthMiddleware``. Without this, every new login after any password
    change would be bounced straight back to ``/login`` (the version stored
    in the DB would never match the ``None`` that a brand-new session starts
    with).
    """
    # Import locally to avoid a module-level cycle: webgui.state imports
    # webgui.auth (lazily, inside a function) during admin-user bootstrap.
    from webgui.state import state

    app.storage.user["authenticated"] = True
    app.storage.user["username"] = username
    current_version = await state.db.get_setting(SESSION_VERSION_KEY)
    app.storage.user["session_version"] = current_version


def clear_authentication() -> None:
    """Clear authentication from the current browser session."""
    app.storage.user["authenticated"] = False
    app.storage.user.pop("username", None)
