"""Entry point for the NiceGUI-based Smart Albums web interface.

Run with:
    python -m webgui.main

Or from the project root:
    .venv/bin/python -m webgui.main
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Request
from nicegui import Client, app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, Response

from webgui.state import init_state, shutdown_state, state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@app.on_startup
async def startup() -> None:
    """Initialize application state on server start."""
    logger.info("Starting Smart Albums WebGUI...")
    await init_state()
    from webgui.services.scheduler import start_scheduler
    start_scheduler()

    # Warn prominently if the first-time setup hasn't been completed and
    # the server is publicly reachable (binding to 0.0.0.0).
    host = os.environ.get("HOST", "0.0.0.0")
    if state.config.auth.enabled:
        user_count = await state.db.user_count()
        if user_count == 0 and host == "0.0.0.0":
            logger.warning(
                "⚠️  No admin account exists yet and the server is bound to "
                "0.0.0.0 (all interfaces). The first visitor to /setup will "
                "become the admin. Set ADMIN_USERNAME and ADMIN_PASSWORD env "
                "vars, or complete setup immediately after starting the server."
            )

    logger.info("Startup complete.")


@app.on_shutdown
async def shutdown() -> None:
    """Clean up resources on server shutdown."""
    logger.info("Shutting down...")
    from webgui.services.scheduler import stop_scheduler
    stop_scheduler()
    await shutdown_state()


# Import pages to register their routes with NiceGUI
import webgui.pages.login  # noqa: E402, F401
import webgui.pages.dashboard  # noqa: E402, F401
import webgui.pages.settings  # noqa: E402, F401
import webgui.pages.prompts  # noqa: E402, F401
import webgui.pages.new_job  # noqa: E402, F401
import webgui.pages.jobs  # noqa: E402, F401
import webgui.pages.account  # noqa: E402, F401
import webgui.pages.schedules  # noqa: E402, F401
import webgui.pages.user_pipelines  # noqa: E402, F401
import webgui.pages.tools  # noqa: E402, F401


# Page routes that do not require authentication.
UNRESTRICTED_ROUTES = frozenset({"/favicon.ico", "/login", "/setup"})


class AuthMiddleware(BaseHTTPMiddleware):
    """HTTP-level authentication gate.

    Performing redirects here (rather than via ``ui.navigate.to`` inside a
    page handler) guarantees the browser is redirected before any page is
    rendered. In-page redirects during page load are unreliable in NiceGUI
    and can leave the user staring at a blank page.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        # Handle logout at the HTTP level so it always redirects cleanly,
        # regardless of whether auth is currently enabled.
        if path == "/logout":
            app.storage.user["authenticated"] = False
            app.storage.user.pop("username", None)
            return RedirectResponse("/login")

        # When auth is disabled, everything else is public.
        if not state.config.auth.enabled:
            return await call_next(request)

        # Never gate NiceGUI's own static assets or websocket endpoints.
        if path.startswith("/_nicegui"):
            return await call_next(request)

        # Only gate real page routes; let everything else pass.
        if path in Client.page_routes.values():
            authenticated = app.storage.user.get("authenticated", False)

            # Validate session version — password changes bump this to
            # invalidate all other active sessions.
            if authenticated:
                session_ver = app.storage.user.get("session_version")
                current_ver = await state.db.get_setting("session_version")
                if current_ver is not None and session_ver != current_ver:
                    # Session is stale — force re-login
                    app.storage.user["authenticated"] = False
                    app.storage.user.pop("username", None)
                    authenticated = False

            if authenticated and path in UNRESTRICTED_ROUTES:
                # Already logged in — no reason to show login/setup.
                logger.info("Redirecting %s -> /", path)
                return RedirectResponse("/", status_code=303)
            if not authenticated and path not in UNRESTRICTED_ROUTES:
                logger.info("Redirecting %s -> /login", path)
                return RedirectResponse(f"/login?redirect_to={path}", status_code=303)

        return await call_next(request)


app.add_middleware(AuthMiddleware)


def _get_storage_secret() -> str:
    """Get or generate the storage secret for NiceGUI sessions.

    Reads from STORAGE_SECRET env var. If not set, generates a random
    secret (sessions won't persist across server restarts in that case).
    """
    secret = os.environ.get("STORAGE_SECRET")
    if secret:
        return secret
    logger.warning(
        "STORAGE_SECRET not set — generating random secret. "
        "Sessions will not persist across restarts."
    )
    return secrets.token_urlsafe(32)


def main() -> None:
    """Run the NiceGUI application."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    ui.run(
        title="Smart Albums",
        host=host,
        port=port,
        dark=True,
        storage_secret=_get_storage_secret(),
        show=False,
        reload=os.environ.get("DEV", "").lower() in ("1", "true"),
    )


if __name__ == "__main__":
    main()
