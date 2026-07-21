"""Rich-based terminal progress reporter."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn


class RichProgressReporter:
    """Terminal progress reporter using a single persistent Rich Progress bar.

    Avoids visual flickering by reusing one Progress widget for the entire
    pipeline run. Each stage gets its own task within the persistent bar;
    previous tasks are hidden (made invisible) when a new stage begins.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._progress: Progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._task_id: TaskID | None = None
        self._started: bool = False

    def start_stage(self, description: str, total: int | None = None) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

        # Hide the previous task if there was one
        if self._task_id is not None:
            self._progress.update(self._task_id, visible=False)

        self._task_id = self._progress.add_task(description, total=total or 0)

    def advance(self, n: int = 1) -> None:
        if self._task_id is not None:
            self._progress.advance(self._task_id, advance=n)

    def finish_stage(self) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, visible=False)
            self._task_id = None

    def log(self, message: str) -> None:
        if self._started:
            self._progress.console.print(message)
        else:
            self._console.print(message)

    def warn(self, message: str) -> None:
        if self._started:
            self._progress.console.print(f"[yellow]⚠ {message}[/yellow]")
        else:
            self._console.print(f"[yellow]⚠ {message}[/yellow]")

    def stop(self) -> None:
        """Stop the progress display. Called at pipeline end."""
        if self._started:
            self._progress.stop()
            self._started = False
