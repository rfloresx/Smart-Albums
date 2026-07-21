"""No-op progress reporter for tests and headless environments."""

from __future__ import annotations


class NullProgressReporter:
    """No-op progress reporter for tests and headless environments."""

    def start_stage(self, description: str, total: int | None = None) -> None:
        pass

    def advance(self, n: int = 1) -> None:
        pass

    def finish_stage(self) -> None:
        pass

    def log(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass
