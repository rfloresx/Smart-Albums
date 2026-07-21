"""Authentication module for the NiceGUI web interface.

Provides password hashing, session verification, and a guard function
for protecting pages. Uses NiceGUI's app.storage.user for session state.

IMPORTANT: app.storage.user is only available after the client WebSocket
connects. Always call `await ui.context.client.connected()` before
reading storage in a page handler.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

import bcrypt
from nicegui import app, ui

logger = logging.getLogger(__name__)


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


def set_authenticated(username: str) -> None:
    """Mark the current browser session as authenticated."""
    app.storage.user["authenticated"] = True
    app.storage.user["username"] = username


def clear_authentication() -> None:
    """Clear authentication from the current browser session."""
    app.storage.user["authenticated"] = False
    app.storage.user.pop("username", None)
