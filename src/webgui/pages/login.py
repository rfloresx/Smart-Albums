"""Login and first-time setup page for the NiceGUI web interface."""

from __future__ import annotations

import logging
import uuid
import asyncio

from nicegui import app, ui

from webgui.auth import (
    hash_password,
    set_authenticated,
    verify_password,
)
from webgui.state import state

logger = logging.getLogger(__name__)


@ui.page("/login")
async def login_page() -> None:
    """Login page — shows setup if no users, login form otherwise.

    The authentication gate (redirecting logged-in users away, or
    unauthenticated users here) is handled by ``AuthMiddleware`` at the
    HTTP level, so this handler only renders the appropriate form.
    """
    # Wait for client connection so app.storage.user is available
    await ui.context.client.connected()

    # Check if setup is needed (no users exist)
    user_count = await state.db.user_count()
    if user_count == 0:
        _build_setup_ui()
    else:
        _build_login_ui()
    return None


def _build_login_ui() -> None:
    """Render the login form UI."""
    ui.dark_mode(True)

    with ui.column().classes("absolute-center items-center"):
        with ui.card().classes("w-80"):
            ui.label("📷 Smart Albums").classes(
                "text-h5 text-center w-full q-mb-md"
            )

            username_input = ui.input(
                label="Username",
                placeholder="Enter username",
            ).classes("w-full").props("autofocus")

            password_input = ui.input(
                label="Password",
                password=True,
                password_toggle_button=True,
                placeholder="Enter password",
            ).classes("w-full")

            error_label = ui.label("").classes(
                "text-negative text-center w-full"
            )
            error_label.set_visibility(False)

            async def handle_login() -> None:
                username = username_input.value.strip()
                password = password_input.value

                if not username or not password:
                    error_label.text = (
                        "Username and password are required"
                    )
                    error_label.set_visibility(True)
                    return

                user = await state.db.get_user(username)
                if user is None or not verify_password(
                    password, user["password_hash"]
                ):
                    error_label.text = "Invalid username or password"
                    error_label.set_visibility(True)
                    return

                set_authenticated(username)
                logger.info("User logged in: %s", username)
                await asyncio.sleep(0)
                ui.navigate.to("/")

            ui.button(
                "Log in", on_click=handle_login
            ).classes("w-full q-mt-md").props("color=primary")

            password_input.on("keydown.enter", handler=handle_login)


def _build_setup_ui() -> None:
    """Render the first-time setup form UI."""
    ui.dark_mode(True)

    with ui.column().classes("absolute-center items-center"):
        with ui.card().classes("w-80"):
            ui.label("📷 Smart Albums — Setup").classes(
                "text-h5 text-center w-full q-mb-sm"
            )
            ui.label(
                "Create your admin account to get started."
            ).classes("text-center w-full q-mb-md text-grey")

            username_input = ui.input(
                label="Username",
                value="admin",
            ).classes("w-full")

            password_input = ui.input(
                label="Password",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            confirm_input = ui.input(
                label="Confirm Password",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            error_label = ui.label("").classes(
                "text-negative text-center w-full"
            )
            error_label.set_visibility(False)

            async def handle_setup() -> None:
                username = username_input.value.strip()
                password = password_input.value
                confirm = confirm_input.value

                if not username:
                    error_label.text = "Username is required"
                    error_label.set_visibility(True)
                    return

                if len(password) < 8:
                    error_label.text = (
                        "Password must be at least 8 characters"
                    )
                    error_label.set_visibility(True)
                    return

                if password != confirm:
                    error_label.text = "Passwords do not match"
                    error_label.set_visibility(True)
                    return

                # Double-check no users were created in the meantime
                count = await state.db.user_count()
                if count > 0:
                    ui.navigate.to("/login")
                    return

                pw_hash = hash_password(password)
                await state.db.create_user(
                    str(uuid.uuid4()), username, pw_hash
                )
                logger.info(
                    "Initial admin account created via setup: %s",
                    username,
                )

                set_authenticated(username)
                ui.navigate.to("/")

            ui.button(
                "Create Account", on_click=handle_setup
            ).classes("w-full q-mt-md").props("color=primary")

            confirm_input.on("keydown.enter", handler=handle_setup)


@ui.page("/setup")
async def setup_page() -> None:
    """Direct access to setup page.

    Renders the setup form when no users exist, otherwise the login form.
    Logout and the authenticated/unauthenticated gate are handled by
    ``AuthMiddleware``.
    """
    await ui.context.client.connected()

    user_count = await state.db.user_count()
    if user_count > 0:
        _build_login_ui()
    else:
        _build_setup_ui()
