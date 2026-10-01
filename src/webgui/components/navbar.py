"""Persistent navigation bar rendered on all authenticated pages.

Layout:
    [Icon/Logo] | [Pipeline] | [Jobs] | [Prompts] |    [☁✓]   | [Username ▾]

The cloud icon summarizes provider health (green=all ok, yellow=partial,
red=all down). Clicking opens a dropdown with per-provider status.

The username dropdown provides access to Provider Settings, Account, and Logout.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nicegui import ui

from webgui.auth import get_current_username

logger = logging.getLogger(__name__)

# --- Shared provider-status cache -------------------------------------------
#
# Each open browser tab renders its own navbar, and previously each one
# started its own `ui.timer(30.0, refresh_status)` that independently called
# every configured provider's health check endpoint. With N tabs open that's
# N x (number of providers) outbound health-check calls every 30 seconds
# for data that's identical across tabs (WG-22). Instead, a single
# process-wide background task refreshes `_latest_status` on an interval,
# and each tab's navbar just reads the cached value and re-renders — no
# per-tab network calls.
_latest_status: dict[str, tuple[bool | None, str, str]] = {}
_status_ready = asyncio.Event()
_background_task: asyncio.Task[None] | None = None
_STATUS_REFRESH_INTERVAL = 30.0


def build_navbar() -> None:
    """Render the application-wide navigation bar.

    Call this at the top of every authenticated page handler (after
    ``ui.dark_mode(True)``). It uses ``ui.header()`` which sticks to the
    top of the viewport across scrolls.
    """
    username = get_current_username() or "User"

    with ui.header().classes("items-center justify-between q-px-md"):
        # Left section: logo + nav links
        with ui.row().classes("items-center gap-1"):
            # App icon / logo
            ui.button(
                "📷 SmartAlbums",
                on_click=lambda: ui.navigate.to("/"),
            ).props("flat color=white no-caps").classes("text-subtitle1")

            ui.separator().props("vertical").classes("q-mx-sm")

            # Primary nav items
            ui.button(
                "Pipeline",
                icon="rocket_launch",
                on_click=lambda: ui.navigate.to("/jobs/new"),
            ).props("flat color=white no-caps size=md")

            ui.button(
                "Jobs",
                icon="history",
                on_click=lambda: ui.navigate.to("/jobs"),
            ).props("flat color=white no-caps size=md")

            ui.button(
                "Schedules",
                icon="schedule",
                on_click=lambda: ui.navigate.to("/schedules"),
            ).props("flat color=white no-caps size=md")

            ui.button(
                "Custom Pipelines",
                icon="account_tree",
                on_click=lambda: ui.navigate.to("/pipelines"),
            ).props("flat color=white no-caps size=md")

            ui.button(
                "Prompts",
                icon="description",
                on_click=lambda: ui.navigate.to("/prompts"),
            ).props("flat color=white no-caps size=md")

        # Right section: status indicator + user menu
        with ui.row().classes("items-center gap-2"):
            # --- Provider status indicator ---
            _build_status_indicator()

            # --- User dropdown ---
            with ui.button(username, icon="person").props(
                "flat color=white no-caps size=md"
            ):
                with ui.menu():
                    ui.menu_item(
                        "Provider Settings",
                        on_click=lambda: ui.navigate.to("/settings"),
                    ).props("dense")
                    ui.menu_item(
                        "Account",
                        on_click=lambda: ui.navigate.to("/account"),
                    ).props("dense")
                    ui.separator()
                    ui.menu_item(
                        "Tools",
                        on_click=lambda: ui.navigate.to("/tools"),
                    ).props("dense")
                    ui.separator()
                    ui.menu_item(
                        "Logout",
                        on_click=lambda: ui.navigate.to("/logout"),
                    ).props("dense")


def _ensure_background_refresh() -> None:
    """Start the shared provider-health polling task if it isn't running yet.

    Safe to call from every tab's navbar — only the first call actually
    starts the task (WG-22).
    """
    global _background_task
    if _background_task is not None and not _background_task.done():
        return

    async def _loop() -> None:
        while True:
            try:
                global _latest_status
                _latest_status = await _check_all_providers()
            except Exception:  # noqa: BLE001
                logger.exception("Provider status refresh failed")
            finally:
                _status_ready.set()
            await asyncio.sleep(_STATUS_REFRESH_INTERVAL)

    _background_task = asyncio.ensure_future(_loop())


def _build_status_indicator() -> None:
    """Build the cloud status icon with a dropdown menu showing per-provider health.

    The icon starts grey (unknown) and updates once the shared background
    refresh (see `_ensure_background_refresh`) has a result. Each tab polls
    the shared `_latest_status` cache locally every few seconds to redraw —
    it does not trigger its own health-check calls (WG-22).
    """
    _ensure_background_refresh()

    # The button that shows the summary icon
    status_btn = ui.button(icon="cloud_queue").props(
        "flat round dense size=sm color=grey-5"
    ).tooltip("Provider status — loading…")

    # The dropdown menu (populated after status fetch)
    menu = ui.menu().props("anchor='bottom right' self='top right'")
    status_btn.on("click", lambda: menu.open())

    # Container inside the menu for status items
    with menu:
        status_container = ui.column().classes("q-pa-sm gap-1").style("min-width: 200px")

    last_rendered: dict[str, Any] = {"results": None}

    def render_from_cache() -> None:
        if _latest_status == last_rendered["results"]:
            return
        last_rendered["results"] = dict(_latest_status)
        _update_indicator(status_btn, status_container, _latest_status)

    if _status_ready.is_set():
        render_from_cache()

    # Light local timer just re-renders this tab's icon from the shared
    # cache — it never performs a network call itself.
    ui.timer(2.0, render_from_cache)


def _update_indicator(
    btn: Any,
    container: Any,
    results: dict[str, tuple[bool | None, str, str]],
) -> None:
    """Update the status button icon/color and dropdown content.

    Args:
        btn: The status button element.
        container: The column inside the dropdown menu.
        results: Dict of slot_name -> (connected, provider_name, message).
            connected is True/False/None (None = not configured or no health check).
    """
    # Determine summary state
    statuses = [v[0] for v in results.values()]
    configured = [s for s in statuses if s is not None]

    if not configured:
        icon = "cloud_off"
        color = "grey-5"
        tooltip = "No providers configured"
    elif all(configured):
        icon = "cloud_done"
        color = "positive"
        tooltip = "All providers connected"
    elif any(configured):
        icon = "cloud_queue"
        color = "warning"
        tooltip = "Some providers unreachable"
    else:
        icon = "cloud_off"
        color = "negative"
        tooltip = "All providers unreachable"

    btn.props(f"icon={icon} color={color}")
    btn._props["icon"] = icon
    btn.tooltip(tooltip)
    btn.update()

    # Rebuild dropdown content
    container.clear()
    with container:
        ui.label("Provider Status").classes("text-caption text-weight-bold q-mb-xs")
        if not results:
            ui.label("No health-checkable providers configured").classes(
                "text-caption text-grey"
            )
        for slot_name, (connected, provider_name, message) in results.items():
            label = _get_slot_label(slot_name)
            _render_status_row(label, provider_name, connected, message)


def _render_status_row(
    label: str,
    provider_name: str,
    connected: bool | None,
    message: str,
) -> None:
    """Render a single provider status row in the dropdown."""
    if connected is None:
        dot_color = "grey"
        status_text = "Not configured"
    elif connected:
        dot_color = "positive"
        status_text = provider_name or "Connected"
    else:
        dot_color = "negative"
        status_text = message or "Unreachable"

    with ui.row().classes("items-center gap-2 no-wrap"):
        ui.icon("circle", size="10px", color=dot_color)
        ui.label(f"{label}:").classes("text-caption text-weight-medium").style(
            "min-width: 70px"
        )
        ui.label(status_text).classes("text-caption text-grey")


def _get_slot_label(slot_name: str) -> str:
    """Get the human-friendly label for a slot by looking it up dynamically."""
    from webgui.services import providers as providers_service

    for slot in providers_service.PROTOCOL_SLOTS:
        if slot.name == slot_name:
            return slot.label
    return slot_name.title()


async def _check_all_providers() -> dict[str, tuple[bool | None, str, str]]:
    """Check health of all configured provider slots that support IHealthCheck.

    Dynamically discovers which slots have a selected provider that implements
    IHealthCheck, and runs health checks only for those.

    Returns:
        Dict mapping slot name to (connected, provider_name, message).
        ``connected`` is None when the slot has no provider selected.
    """
    from webgui.services import providers as providers_service
    from webgui.state import state

    results: dict[str, tuple[bool | None, str, str]] = {}

    for slot in providers_service.PROTOCOL_SLOTS:
        prov_cfg = state.config.get_provider(slot.name)
        if not prov_cfg.selected:
            continue

        # Check if the selected provider supports health check
        provider_specs = providers_service.get_providers_for(slot)
        provider_spec = next(
            (p for p in provider_specs if p.name == prov_cfg.selected), None
        )
        if provider_spec is None or not provider_spec.supports_health_check:
            continue

        try:
            ok, message = await asyncio.wait_for(
                providers_service.test_connection(
                    slot, prov_cfg.selected, prov_cfg.params, timeout=10.0
                ),
                timeout=12.0,
            )
            results[slot.name] = (ok, prov_cfg.selected, message)
        except asyncio.TimeoutError:
            results[slot.name] = (False, prov_cfg.selected, "Timed out")
        except Exception as exc:  # noqa: BLE001
            results[slot.name] = (False, prov_cfg.selected, str(exc))

    return results
