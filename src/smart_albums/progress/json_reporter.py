"""JSON-line progress reporter for machine-readable pipeline progress.

Emits structured progress events as JSON lines to a writable stream
(typically stdout). Designed for subprocess-based execution where a
parent process (e.g., the web GUI) parses progress from the output.

Each line is prefixed with a magic marker so the consumer can distinguish
progress events from regular log output.
"""

from __future__ import annotations

import json
import sys
import time
from typing import IO, Any


# Prefix that identifies a line as a progress event
PROGRESS_PREFIX = "@@PROGRESS@@"


class JsonProgressReporter:
    """Emits JSON progress events to a writable stream.

    Progress protocol:
    - ``start_stage(description, total)`` → emits stage_start event
    - ``advance(n)`` → emits advance event with current/total counts
    - ``finish_stage()`` → emits stage_finish event
    - ``log(message)`` → emits log event
    - ``warn(message)`` → emits warn event

    Pipeline-level tracking:
    - ``start_pipeline(total_stages)`` → emits pipeline_start event
    - ``set_stage_index(index)`` → sets current stage index for context

    Advance events are throttled to avoid flooding the output stream.
    At most one advance event is emitted per second or per 2% of total.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout
        self._current_stage: str | None = None
        self._total: int = 0
        self._completed: int = 0
        self._stage_index: int = 0
        self._total_stages: int = 0
        self._last_advance_time: float = 0.0
        self._last_advance_completed: int = 0

    def _emit(self, event: dict[str, Any]) -> None:
        """Write a JSON event line to the stream."""
        line = f"{PROGRESS_PREFIX}{json.dumps(event)}"
        self._stream.write(line + "\n")
        self._stream.flush()

    # ------------------------------------------------------------------
    # Pipeline-level methods (called by the runner)
    # ------------------------------------------------------------------

    def start_pipeline(self, total_stages: int) -> None:
        """Signal that the pipeline is starting with a known number of stages."""
        self._total_stages = total_stages
        self._stage_index = 0
        self._emit({
            "type": "pipeline_start",
            "total_stages": total_stages,
        })

    def set_stage_index(self, index: int, name: str) -> None:
        """Set the current stage index (0-based) within the pipeline."""
        self._stage_index = index
        self._emit({
            "type": "stage_enter",
            "stage_index": index,
            "total_stages": self._total_stages,
            "stage_name": name,
        })

    # ------------------------------------------------------------------
    # IProgressReporter interface
    # ------------------------------------------------------------------

    def start_stage(self, description: str, total: int | None = None) -> None:
        """Signal that a stage is starting per-asset processing."""
        self._current_stage = description
        self._total = total or 0
        self._completed = 0
        self._last_advance_time = time.monotonic()
        self._last_advance_completed = 0
        self._emit({
            "type": "stage_start",
            "stage": description,
            "stage_index": self._stage_index,
            "total_stages": self._total_stages,
            "total": self._total,
        })

    def advance(self, n: int = 1) -> None:
        """Record progress of n items within the current stage.

        Throttled: emits at most once per second or per 2% progress.
        """
        self._completed += n

        # Throttle: emit if >=1s elapsed or >=2% progress since last emit
        now = time.monotonic()
        elapsed = now - self._last_advance_time
        pct_delta = 0.0
        if self._total > 0:
            pct_delta = (self._completed - self._last_advance_completed) / self._total

        if elapsed >= 1.0 or pct_delta >= 0.02 or self._completed >= self._total:
            self._last_advance_time = now
            self._last_advance_completed = self._completed
            self._emit({
                "type": "advance",
                "stage": self._current_stage,
                "stage_index": self._stage_index,
                "total_stages": self._total_stages,
                "completed": self._completed,
                "total": self._total,
            })

    def finish_stage(self) -> None:
        """Signal that the current stage has finished."""
        self._emit({
            "type": "stage_finish",
            "stage": self._current_stage,
            "stage_index": self._stage_index,
            "total_stages": self._total_stages,
            "completed": self._completed,
            "total": self._total,
        })
        self._current_stage = None
        self._total = 0
        self._completed = 0

    def log(self, message: str) -> None:
        """Emit an informational log message."""
        self._emit({
            "type": "log",
            "message": message,
        })

    def warn(self, message: str) -> None:
        """Emit a warning message."""
        self._emit({
            "type": "warn",
            "message": message,
        })
