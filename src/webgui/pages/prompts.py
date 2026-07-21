"""Prompt management page — CRUD UI for scoring prompt files.

Prompts are rendered as a card grid. Creating and editing both use the same
dialog (with content textarea); deleting asks for confirmation first. All
mutations go through ``webgui.services.prompts``, which owns filename safety
and persistence to disk.
"""

from __future__ import annotations

import logging

from nicegui import ui

from webgui.components.navbar import build_navbar
from webgui.services import prompts as prompts_service

logger = logging.getLogger(__name__)


def _build_header() -> None:
    """Render the shared navigation bar."""
    build_navbar()


def _preview(content: str, max_chars: int = 220) -> str:
    """Shorten prompt content for the card preview."""
    text = content.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


class _PromptEditor:
    """Shared create/edit dialog for a prompt file.

    In create mode the name field is editable; in edit mode the name is
    fixed (renaming isn't supported — delete and recreate instead) and only
    the content can change.
    """

    def __init__(self, on_saved) -> None:
        self._on_saved = on_saved
        self._editing_name: str | None = None

        with ui.dialog() as self.dialog, ui.card().classes("w-full max-w-2xl"):
            self.title_label = ui.label("New Prompt").classes("text-h6")
            self.name_input = ui.input(label="Name").classes("w-full").props(
                "outlined dense"
            )
            self.content_input = ui.textarea(label="Prompt content").classes(
                "w-full"
            ).props("outlined rows=16")
            self.error_label = ui.label("").classes("text-negative")
            self.error_label.set_visibility(False)

            with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
                ui.button("Cancel", on_click=self.dialog.close).props("flat")
                ui.button("Save", on_click=self._save).props("color=primary")

    def open_create(self) -> None:
        self._editing_name = None
        self.title_label.text = "New Prompt"
        self.name_input.value = ""
        self.name_input.set_visibility(True)
        self.content_input.value = ""
        self.error_label.set_visibility(False)
        self.dialog.open()

    def open_edit(self, prompt: prompts_service.PromptInfo) -> None:
        self._editing_name = prompt.name
        self.title_label.text = f"Edit — {prompt.label}"
        self.name_input.value = prompt.name
        self.name_input.set_visibility(False)
        self.content_input.value = prompt.content
        self.error_label.set_visibility(False)
        self.dialog.open()

    async def _save(self) -> None:
        content = self.content_input.value or ""
        try:
            if self._editing_name is None:
                prompts_service.create_prompt(self.name_input.value or "", content)
            else:
                prompts_service.update_prompt(self._editing_name, content)
        except prompts_service.PromptError as exc:
            self.error_label.text = str(exc)
            self.error_label.set_visibility(True)
            return

        self.dialog.close()
        await self._on_saved()


@ui.page("/prompts")
async def prompts_page() -> None:
    """Prompt management page.

    Authentication is enforced by ``AuthMiddleware`` at the HTTP level, so
    reaching this handler means the request is already authorized.
    """
    await ui.context.client.connected()
    ui.dark_mode(True)
    _build_header()

    with ui.column().classes("w-full max-w-5xl mx-auto q-pa-md gap-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Scoring Prompts").classes("text-h4")
            new_btn = ui.button("New prompt", icon="add").props("color=primary")

        ui.label(
            "Prompts are Markdown files sent to the vision model during "
            "aesthetic scoring. Reference a prompt by filename from the "
            "score stage's prompt_file setting."
        ).classes("text-caption text-grey")

        grid = ui.grid(columns=3).classes("w-full gap-4")

        async def refresh() -> None:
            grid.clear()
            items = prompts_service.list_prompts()
            with grid:
                if not items:
                    ui.label("No prompts yet — create one to get started.").classes(
                        "text-caption text-grey col-span-3"
                    )
                for item in items:
                    _render_card(item)

        def _render_card(item: prompts_service.PromptInfo) -> None:
            with ui.card().classes("w-full"):
                ui.label(item.label).classes("text-subtitle1")
                ui.label(item.name).classes("text-caption text-grey")
                ui.label(_preview(item.content)).classes("text-body2 q-mt-sm")
                with ui.row().classes("w-full justify-end gap-1 q-mt-sm"):
                    ui.button(
                        icon="edit", on_click=lambda i=item: editor.open_edit(i)
                    ).props("flat dense round size=sm")
                    ui.button(
                        icon="delete",
                        on_click=lambda i=item: confirm_delete(i),
                    ).props("flat dense round size=sm color=negative")

        def confirm_delete(item: prompts_service.PromptInfo) -> None:
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Delete {item.label!r}?").classes("text-h6")
                ui.label(
                    "This permanently removes the prompt file. Pipelines "
                    "referencing it by filename will fail until reconfigured."
                ).classes("text-caption text-grey")

                async def do_delete() -> None:
                    try:
                        prompts_service.delete_prompt(item.name)
                    except prompts_service.PromptError as exc:
                        ui.notify(str(exc), type="negative")
                    else:
                        ui.notify(f"Deleted {item.name}", type="positive")
                    dialog.close()
                    await refresh()

                with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Delete", on_click=do_delete).props(
                        "color=negative"
                    )
            dialog.open()

        editor = _PromptEditor(on_saved=refresh)
        new_btn.on_click(editor.open_create)

        await refresh()
