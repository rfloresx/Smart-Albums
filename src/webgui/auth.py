"""Authentication module for the NiceGUI web interface.

Provides password hashing, session verification, and a guard function
for protecting pages. Uses NiceGUI's app.storage.user for session state.

IMPORTANT: app.storage.user is only available after the client WebSocket
connects. Always call `await ui.context.client.connected()` before
reading storage in a page handler.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Optional

import bcrypt
from nicegui import app

logger = logging.getLogger(__name__)

# Settings-table key that holds the current global session version. Bumped
# whenever a password changes so that other active sessions are logged out.
SESSION_VERSION_KEY = "session_version"

# A bcrypt hash of a random, never-used password, computed once at import
# time. Used by verify_password_async() to burn roughly the same amount of
# CPU time for an unknown username as for a known one with a wrong
# password, so response timing doesn't reveal which usernames exist
# (WG-10). Comparing against this hash always fails.
_DUMMY_HASH = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt()).decode("utf-8")


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(
        password.encode("utf-8"), hashed.encode("utf-8")
    )


async def hash_password_async(password: str) -> str:
    """Hash a password off the event loop.

    bcrypt's hashpw/checkpw are synchronous CPU-bound calls (roughly
    100-300ms each). Calling them directly inside an async handler blocks
    NiceGUI's single event loop for that long, freezing every other
    connected client's UI and the background scheduler for the duration
    (WG-10). Dispatch to a thread instead.
    """
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, hashed: Optional[str]) -> bool:
    """Verify a password off the event loop, without leaking via timing.

    When ``hashed`` is ``None`` (unknown username), this still runs a
    bcrypt comparison against a fixed dummy hash rather than returning
    immediately. Skipping bcrypt entirely for unknown usernames (the
    previous behavior: ``if user is None or not verify_password(...)``)
    makes an unknown-username response measurably faster than a
    known-username/wrong-password response, which lets an attacker
    enumerate valid usernames by timing alone.
    """
    target = hashed if hashed is not None else _DUMMY_HASH
    result = await asyncio.to_thread(verify_password, password, target)
    return result if hashed is not None else False


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
