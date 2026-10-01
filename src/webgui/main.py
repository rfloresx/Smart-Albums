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
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, Response

from webgui.auth import SESSION_VERSION_KEY
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

    # If the server is publicly reachable (binding to 0.0.0.0) and no admin
    # account exists yet, the first visitor to /setup would otherwise become
    # the admin. Refuse to serve a public, unauthenticated setup form: either
    # ADMIN_USERNAME/ADMIN_PASSWORD must be set (state.init() already seeds
    # the admin from them), or the operator must explicitly opt in via
    # ALLOW_OPEN_SETUP=1 (e.g. for a first-run behind a firewall/VPN).
    host = os.environ.get("HOST", "0.0.0.0")
    if state.config.auth.enabled:
        user_count = await state.db.user_count()
        if user_count == 0 and host == "0.0.0.0":
            allow_open_setup = os.environ.get("ALLOW_OPEN_SETUP", "").lower() in ("1", "true")
            if not allow_open_setup:
                # NiceGUI's @app.on_startup hook logs an exception raised
                # here but does NOT stop the server -- the process keeps
                # running and /setup stays reachable, which would defeat the
                # whole point of this check. Exit the process directly.
                logger.error(
                    "No admin account exists and the server is bound to 0.0.0.0 "
                    "(all interfaces). Refusing to start with an open, "
                    "unauthenticated /setup page. Set ADMIN_USERNAME and "
                    "ADMIN_PASSWORD env vars to auto-create the admin account, "
                    "bind to 127.0.0.1 instead, or set ALLOW_OPEN_SETUP=1 to "
                    "accept the risk (only the first visitor to /setup will "
                    "become admin)."
                )
                # os._exit rather than sys.exit: this runs inside NiceGUI's
                # own startup task, several layers under the ASGI lifespan
                # handler. sys.exit() there just raises SystemExit that
                # propagates through asyncio/uvloop machinery as a noisy
                # (but harmless) traceback. os._exit terminates the process
                # immediately and unambiguously.
                os._exit(1)
            logger.warning(
                "⚠️  No admin account exists yet and the server is bound to "
                "0.0.0.0 (all interfaces). ALLOW_OPEN_SETUP is set, so the "
                "first visitor to /setup will become the admin."
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


# Page and API routes that do not require authentication.
UNRESTRICTED_ROUTES = frozenset({"/favicon.ico", "/login", "/setup"})

# Path prefixes that are always public (NiceGUI's own static/websocket
# machinery). Everything else -- every page route AND every plain FastAPI
# route such as /api/* -- is gated by default. This is a deny-by-default
# policy: routes are only reachable without authentication if their path is
# explicitly listed above or starts with one of these prefixes.
_PUBLIC_PREFIXES = ("/_nicegui",)


class AuthMiddleware(BaseHTTPMiddleware):
    """HTTP-level authentication gate.

    Performing redirects here (rather than via ``ui.navigate.to`` inside a
    page handler) guarantees the browser is redirected before any page is
    rendered. In-page redirects during page load are unreliable in NiceGUI
    and can leave the user staring at a blank page.

    Unlike an earlier version of this middleware, this gate is deny-by-
    default: it does not only check NiceGUI's ``@ui.page`` routes. Plain
    FastAPI routes registered with ``@app.get``/``@app.post`` (e.g.
    ``/api/tools/backup``, ``/api/jobs/{id}/download``) are gated the same
    way, since they carry no less sensitive data than a page.
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
        if path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        authenticated = app.storage.user.get("authenticated", False)

        # Validate session version — password changes bump this to
        # invalidate all other active sessions.
        if authenticated:
            session_ver = app.storage.user.get("session_version")
            current_ver = await state.db.get_setting(SESSION_VERSION_KEY)
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
            # Deny-by-default: this covers both @ui.page routes and plain
            # FastAPI routes (e.g. /api/*). API-style requests (JSON,
            # fetch/XHR, or paths under /api/) get a 401 instead of a
            # redirect, since a redirect to an HTML login page is not a
            # useful response for a download endpoint or a script.
            if path.startswith("/api/") or "text/html" not in request.headers.get("accept", ""):
                logger.info("Rejecting unauthenticated request to %s", path)
                return Response(status_code=401, content="Unauthorized")
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
