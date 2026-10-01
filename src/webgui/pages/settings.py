"""Admin provider configuration page.

Renders a dynamic form for each configurable protocol slot (image library,
vision/LLM, embedding, cache). For each slot the admin picks a provider and
fills in its parameters — the parameter set is introspected from the provider
class, so adding a new provider requires no UI changes.

Each slot supports an inline "Test connection" action (for providers that
implement ``IHealthCheck``). Saving writes the selections and parameters back
to the database and reloads the in-memory config.

Also includes authentication settings (enable/disable, session lifetime).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from nicegui import ui

from webgui.auth import get_current_username
from webgui.components.navbar import build_navbar
from webgui.config import AuthConfig, ProviderConfig
from webgui.services import providers as providers_service
from webgui.state import state

logger = logging.getLogger(__name__)


def _prettify(name: str) -> str:
    """Turn a snake_case parameter name into a Title Case label."""
    return name.replace("_", " ").strip().title()


# Sentinel placeholder shown for a previously-saved secret instead of its
# real value. "Collect" treats a field left at exactly this placeholder as
# "keep the existing stored value", so a save that doesn't touch the secret
# field doesn't need to resend it, and the browser never receives the
# actual key over the websocket or sees it sitting in the page's DOM
# (WG-11). A real secret value is extremely unlikely to collide with this
# literal string, but to be certain, the field will not revert to the
# placeholder even if a user types it manually — only the untouched
# server-rendered default is special-cased in `_coerce`.
_SECRET_PLACEHOLDER = "••••••••(unchanged)"


def _make_field(param: providers_service.ParamSpec, value: Any) -> Any:
    """Create the appropriate NiceGUI input widget for a parameter.

    Args:
        param: The parameter spec describing the widget kind.
        value: The current value to pre-fill (from saved config or default).

    Returns:
        The created NiceGUI element (value read back later via ``.value``).
    """
    label = _prettify(param.name) + ("" if param.required else " (optional)")

    if param.kind == "bool":
        initial = bool(value) if value is not None else bool(param.default)
        return ui.switch(_prettify(param.name), value=initial).classes("w-full")

    if param.kind in ("int", "float"):
        numeric = value if isinstance(value, (int, float)) else param.default
        return ui.number(label=label, value=numeric).classes("w-full")

    if param.kind == "password":
        # Never pre-fill the real secret value into the DOM. If a value is
        # already saved, show a placeholder instead; `_coerce` maps the
        # placeholder back to "keep the stored value" at save time. An
        # empty field (no saved value yet) stays empty as before.
        has_saved_value = bool(value)
        element = ui.input(
            label=label,
            value=_SECRET_PLACEHOLDER if has_saved_value else "",
            password=True,
            password_toggle_button=True,
        ).classes("w-full")
        if has_saved_value:
            element.props("hint='Leave as-is to keep the saved value, or type a new one.'")
        return element

    # Plain string.
    return ui.input(
        label=label,
        value=str(value) if value is not None else "",
    ).classes("w-full")


def _coerce(param: providers_service.ParamSpec, value: Any, saved_value: Any = None) -> Any:
    """Coerce a widget value back to the parameter's expected type.

    For a ``password`` field left at the placeholder (see
    ``_SECRET_PLACEHOLDER``), returns the original saved value instead of
    the placeholder text itself — the user never edited the secret, so the
    stored value should be preserved as-is rather than being overwritten
    with the literal placeholder string.
    """
    if param.kind == "password" and value == _SECRET_PLACEHOLDER:
        return saved_value if saved_value is not None else ""
    if param.kind == "bool":
        return bool(value)
    if param.kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return param.default
    if param.kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return param.default
    return "" if value is None else str(value)


class _ProviderSection:
    """A single card in the settings page for one protocol slot.

    Owns the provider selector, the dynamically rendered parameter fields, and
    an inline connection-test control. The parameter fields re-render whenever
    the selected provider changes.
    """

    def __init__(self, slot: providers_service.ProtocolSlot) -> None:
        self.slot = slot
        self.specs = providers_service.get_providers_for(slot)
        self.spec_by_name = {s.name: s for s in self.specs}
        self.current = state.config.get_provider(slot.name)
        self.fields: dict[str, tuple[providers_service.ParamSpec, Any, Any]] = {}

        names = [s.name for s in self.specs]
        default_sel = (
            self.current.selected
            if self.current.selected in names
            else (names[0] if names else None)
        )

        with ui.card().classes("w-full"):
            ui.label(slot.label).classes("text-h6")
            ui.label(slot.description).classes("text-caption text-grey")

            if not self.specs:
                ui.label("No providers registered.").classes(
                    "text-caption text-warning"
                )
                self.select = None
                return

            self.select = (
                ui.select(names, value=default_sel, label="Provider")
                .classes("w-64")
                .props("outlined dense")
            )

            self.params_box = ui.column().classes("w-full gap-2 q-mt-sm")

            with ui.row().classes("items-center gap-3 q-mt-sm"):
                self.test_btn = ui.button(
                    "Test connection", on_click=self._on_test
                ).props("outline size=sm")
                self.result = ui.label("").classes("text-caption")

            # Model listing button for llm/embedding slots
            if slot.name in ("llm", "embedding"):
                with ui.row().classes("items-center gap-3 q-mt-xs"):
                    ui.button(
                        "List models", icon="format_list_bulleted",
                        on_click=self._on_list_models,
                    ).props("outline size=sm")
                    self.models_container = ui.column().classes("w-full gap-1")
            else:
                self.models_container = None

            self.select.on_value_change(lambda _e: self._render_params())
            self._render_params()

    def _render_params(self) -> None:
        """(Re)render the parameter fields for the currently selected provider."""
        self.params_box.clear()
        self.fields.clear()
        self.result.text = ""

        spec = self.spec_by_name.get(self.select.value) if self.select else None
        if spec is None:
            return

        self.test_btn.set_visibility(spec.supports_health_check)

        # Pre-fill from saved params only when the saved provider is the one
        # currently selected; otherwise fall back to the provider defaults.
        saved = (
            self.current.params
            if self.select.value == self.current.selected
            else {}
        )

        with self.params_box:
            if not spec.params:
                ui.label("No configuration required.").classes(
                    "text-caption text-grey"
                )
            for param in spec.params:
                value = saved.get(param.name, param.default)
                element = _make_field(param, value)
                # Keep the real saved value server-side (not sent to the
                # browser for password fields — see _make_field) so
                # `collect()` can restore it if the user leaves the
                # placeholder untouched.
                self.fields[param.name] = (param, element, value)

    def collect(self) -> tuple[str, dict[str, Any]]:
        """Return the selected provider name and coerced parameter dict."""
        if self.select is None:
            return "", {}
        selected = self.select.value or ""
        params = {
            name: _coerce(param, element.value, saved_value)
            for name, (param, element, saved_value) in self.fields.items()
        }
        return selected, params

    async def _on_test(self) -> None:
        """Run a connection test against the current form values."""
        selected, params = self.collect()
        self.result.text = "Testing…"
        self.result.classes(replace="text-caption text-grey")
        ok, message = await providers_service.test_connection(
            self.slot, selected, params
        )
        self.result.text = message
        self.result.classes(
            replace="text-caption " + ("text-positive" if ok else "text-negative")
        )

    async def _on_list_models(self) -> None:
        """Fetch and display available models from the configured provider."""
        if self.models_container is None:
            return
        self.models_container.clear()
        selected, params = self.collect()
        base_url = (params.get("base_url") or "").rstrip("/")

        if not base_url:
            with self.models_container:
                ui.label("No base_url configured.").classes("text-caption text-warning")
            return

        with self.models_container:
            ui.label("Fetching models…").classes("text-caption text-grey")

        models = await _fetch_models(selected, base_url, params.get("api_key", ""))
        self.models_container.clear()
        with self.models_container:
            if models is None:
                ui.label("Could not fetch models from server.").classes(
                    "text-caption text-negative"
                )
            elif not models:
                ui.label("No models found.").classes("text-caption text-grey")
            else:
                ui.label(f"{len(models)} model(s) available:").classes("text-caption")
                for model in models[:20]:
                    ui.label(f"  • {model}").classes("text-caption text-grey")


def _build_header() -> None:
    """Render the shared navigation bar."""
    build_navbar()


async def _fetch_models(
    provider_name: str, base_url: str, api_key: str = ""
) -> list[str] | None:
    """Fetch available models from a provider's model listing endpoint.

    Supports Ollama (/api/tags) and llamacpp/OpenAI-compatible (/v1/models).
    Returns None on connection failure, empty list if no models found.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            if provider_name == "ollama":
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
            else:
                # llamacpp, openai, or any OpenAI-compatible endpoint
                resp = await client.get(f"{base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        m.get("id", m.get("name", ""))
                        for m in data.get("data", [])
                        if m.get("id") or m.get("name")
                    ]
    except Exception:
        return None
    return None


AUTH_SETTINGS_KEY = "auth"


@ui.page("/settings")
async def settings_page() -> None:
    """Admin provider configuration page.

    Authentication is enforced by ``AuthMiddleware`` at the HTTP level, so
    reaching this handler means the request is already authorized.
    """
    await ui.context.client.connected()
    ui.dark_mode(True)
    _build_header()

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-4"):
        ui.label("Provider Settings").classes("text-h4")
        ui.label(
            "Configure the backends the pipeline uses. Changes are saved to "
            "the database and take effect on the next job run."
        ).classes("text-caption text-grey")

        sections: list[_ProviderSection] = []
        for slot in providers_service.PROTOCOL_SLOTS:
            sections.append(_ProviderSection(slot))

        ui.separator().classes("q-my-md")

        # --- Authentication Settings ---
        ui.label("Authentication").classes("text-h5")
        ui.label(
            "Control login requirements and session behavior. "
            "Disabling authentication makes the UI publicly accessible."
        ).classes("text-caption text-grey")

        with ui.card().classes("w-full"):
            ui.label("🔒 Auth Settings").classes("text-h6")

            was_auth_enabled = state.config.auth.enabled
            auth_enabled_switch = ui.switch(
                "Authentication enabled",
                value=was_auth_enabled,
            )
            ui.label(
                "Disabling this makes every page and API endpoint public, "
                "including this settings page and backup/restore in Tools."
            ).classes("text-caption text-warning")

        ui.separator().classes("q-my-md")

        # --- Save All ---
        async def _do_save() -> None:
            # Save provider config
            updated: dict[str, ProviderConfig] = {}
            for section in sections:
                selected, params = section.collect()
                updated[section.slot.name] = ProviderConfig(
                    selected=selected, params=params
                )
            try:
                await providers_service.save(updated)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to save provider config")
                ui.notify(f"Provider save failed: {exc}", type="negative")
                return

            # Save auth config
            auth_data = {
                "enabled": auth_enabled_switch.value,
            }
            await state.db.set_setting(AUTH_SETTINGS_KEY, auth_data)
            state.config.auth = AuthConfig(**auth_data)

            if was_auth_enabled and not auth_enabled_switch.value:
                logger.warning(
                    "Authentication disabled via settings page by user %r — "
                    "the UI and all /api endpoints are now public.",
                    get_current_username(),
                )

            logger.info("Settings saved (providers + auth)")
            ui.notify("Settings saved", type="positive")

        async def save_all() -> None:
            # Guard against double-submit — otherwise two concurrent clicks
            # could both pass the "was auth enabled" check before either
            # write lands, or fire two overlapping provider saves (WG-23).
            if save_btn.props.get("disable"):
                return
            save_btn.props("disable loading")
            try:
                # Disabling auth makes the whole app public, including this
                # settings page and the Tools page's backup/restore and
                # migration-script runner — require an explicit confirmation
                # rather than letting it happen as a side effect of one click
                # on "Save all" (WG-13).
                if was_auth_enabled and not auth_enabled_switch.value:
                    with ui.dialog() as dlg, ui.card():
                        ui.label("Disable authentication?").classes("text-subtitle1")
                        ui.label(
                            "This makes the entire web UI and all API endpoints "
                            "(including database backup/restore) accessible to "
                            "anyone who can reach this server, with no login "
                            "required. This takes effect immediately for every "
                            "visitor."
                        ).classes("text-body2 text-grey")
                        with ui.row().classes("justify-end w-full gap-2"):
                            ui.button("Cancel", on_click=dlg.close).props("flat")

                            async def _confirm() -> None:
                                dlg.close()
                                await _do_save()

                            ui.button(
                                "Disable authentication and save",
                                color="negative",
                                on_click=_confirm,
                            )
                    dlg.open()
                    return

                await _do_save()
            finally:
                save_btn.props(remove="disable loading")

        with ui.row().classes("w-full justify-end q-mt-md"):
            save_btn = ui.button("Save all", on_click=save_all).props("color=primary")
