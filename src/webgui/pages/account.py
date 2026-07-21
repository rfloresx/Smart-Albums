"""Account management page — password change.

Allows the authenticated user to change their password by providing
their current password and a new one. Validation ensures the current
password is correct and the new password meets minimum length requirements.
"""

from __future__ import annotations

import logging

from nicegui import ui

from webgui.auth import (
    get_current_username,
    hash_password,
    verify_password,
)
from webgui.components.navbar import build_navbar
from webgui.state import state

logger = logging.getLogger(__name__)


@ui.page("/account")
async def account_page() -> None:
    """Account management page — change password.

    Authentication is enforced by ``AuthMiddleware`` at the HTTP level, so
    reaching this handler means the request is already authorized.
    """
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    username = get_current_username()

    with ui.column().classes("w-full max-w-xl mx-auto q-pa-md gap-4"):
        ui.label("Account Settings").classes("text-h4")
        ui.label(f"Logged in as: {username}").classes("text-body1 text-grey")

        ui.separator()

        # --- Change Password ---
        ui.label("Change Password").classes("text-h5 q-mt-md")

        current_pw = ui.input(
            label="Current Password",
            password=True,
            password_toggle_button=True,
        ).classes("w-full").props("outlined dense")

        new_pw = ui.input(
            label="New Password",
            password=True,
            password_toggle_button=True,
        ).classes("w-full").props("outlined dense")

        confirm_pw = ui.input(
            label="Confirm New Password",
            password=True,
            password_toggle_button=True,
        ).classes("w-full").props("outlined dense")

        error_label = ui.label("").classes("text-negative")
        error_label.set_visibility(False)
        success_label = ui.label("").classes("text-positive")
        success_label.set_visibility(False)

        async def change_password() -> None:
            error_label.set_visibility(False)
            success_label.set_visibility(False)

            current = current_pw.value or ""
            new = new_pw.value or ""
            confirm = confirm_pw.value or ""

            if not current:
                error_label.text = "Current password is required."
                error_label.set_visibility(True)
                return

            if len(new) < 8:
                error_label.text = "New password must be at least 8 characters."
                error_label.set_visibility(True)
                return

            if new != confirm:
                error_label.text = "New passwords do not match."
                error_label.set_visibility(True)
                return

            # Verify current password
            user = await state.db.get_user(username)
            if user is None:
                error_label.text = "User not found."
                error_label.set_visibility(True)
                return

            if not verify_password(current, user["password_hash"]):
                error_label.text = "Current password is incorrect."
                error_label.set_visibility(True)
                return

            # Update password
            new_hash = hash_password(new)
            await state.db.update_user_password(username, new_hash)

            # Bump session version to invalidate all other active sessions.
            # The AuthMiddleware/is_authenticated check will reject sessions
            # that have a stale version, forcing re-login.
            import uuid as _uuid
            new_version = str(_uuid.uuid4())[:8]
            await state.db.set_setting("session_version", new_version)
            # Update current session so *this* user stays logged in
            from nicegui import app as _app
            _app.storage.user["session_version"] = new_version

            logger.info("Password changed for user: %s", username)
            current_pw.value = ""
            new_pw.value = ""
            confirm_pw.value = ""
            success_label.text = "Password changed successfully. Other sessions have been invalidated."
            success_label.set_visibility(True)

        ui.button(
            "Change Password", icon="lock", on_click=change_password
        ).props("color=primary")
