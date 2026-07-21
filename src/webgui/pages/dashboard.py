"""Main dashboard page — landing page after login.

Shows quick-action buttons for the most common tasks.
Navigation is handled by the shared navbar component.
"""

from __future__ import annotations

from nicegui import ui

from webgui.components.navbar import build_navbar


@ui.page("/")
async def dashboard_page() -> None:
    """Main application dashboard.

    Authentication is enforced by ``AuthMiddleware`` at the HTTP level, so
    reaching this handler means the request is already authorized.
    """
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    with ui.column().classes("w-full items-center q-pa-lg q-mt-xl"):
        ui.label("Dashboard").classes("text-h4 q-mb-md")
        with ui.row().classes("gap-4"):
            ui.button(
                "New Job",
                icon="play_arrow",
                on_click=lambda: ui.navigate.to("/jobs/new"),
            ).props("color=primary size=lg")
            ui.button(
                "Job History",
                icon="history",
                on_click=lambda: ui.navigate.to("/jobs"),
            ).props("outline size=lg")
            ui.button(
                "Custom Pipelines",
                icon="account_tree",
                on_click=lambda: ui.navigate.to("/pipelines"),
            ).props("outline size=lg")
