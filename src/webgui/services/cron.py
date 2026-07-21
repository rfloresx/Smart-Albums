"""Cron expression parser and next-run calculator.

Supports standard 5-field cron expressions:
    minute hour day_of_month month day_of_week

Field values:
    * — any value
    N — exact value
    N-M — range
    N/S — step (every S starting at N; */S is shorthand for 0/S)
    N,M,... — list

Day of week: 0=Monday .. 6=Sunday (ISO convention).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class CronParseError(ValueError):
    """Raised when a cron expression is invalid."""


def _parse_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching values."""
    values: set[int] = set()

    for part in field.split(","):
        part = part.strip()
        if not part:
            raise CronParseError(f"Empty field component in: {field}")

        # Handle step notation
        step = 1
        if "/" in part:
            base, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                raise CronParseError(f"Invalid step value: {step_str}")
            if step < 1:
                raise CronParseError(f"Step must be >= 1, got: {step}")
            part = base

        if part == "*":
            values.update(range(min_val, max_val + 1, step))
        elif "-" in part:
            # Range
            parts = part.split("-", 1)
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise CronParseError(f"Invalid range: {part}")
            if start < min_val or end > max_val or start > end:
                raise CronParseError(
                    f"Range {start}-{end} out of bounds [{min_val}-{max_val}]"
                )
            values.update(range(start, end + 1, step))
        else:
            # Single value (with possible step applied from the single start)
            try:
                val = int(part)
            except ValueError:
                raise CronParseError(f"Invalid value: {part}")
            if val < min_val or val > max_val:
                raise CronParseError(
                    f"Value {val} out of bounds [{min_val}-{max_val}]"
                )
            if step > 1:
                values.update(range(val, max_val + 1, step))
            else:
                values.add(val)

    return values


class CronExpression:
    """Parsed representation of a 5-field cron expression."""

    def __init__(self, expression: str) -> None:
        self.expression = expression
        parts = expression.strip().split()
        if len(parts) != 5:
            raise CronParseError(
                f"Expected 5 fields (minute hour dom month dow), got {len(parts)}: "
                f"{expression!r}"
            )

        self.minutes = _parse_field(parts[0], 0, 59)
        self.hours = _parse_field(parts[1], 0, 23)
        self.days_of_month = _parse_field(parts[2], 1, 31)
        self.months = _parse_field(parts[3], 1, 12)
        self.days_of_week = _parse_field(parts[4], 0, 6)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        # Python weekday: 0=Monday .. 6=Sunday — same as our convention
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.day in self.days_of_month
            and dt.month in self.months
            and dt.weekday() in self.days_of_week
        )

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next datetime matching this expression after *after*.

        Searches up to 366 days ahead. Raises if no match found (shouldn't
        happen with valid expressions).
        """
        if after is None:
            after = datetime.now(timezone.utc)

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to ~366 days (527040 minutes)
        max_iterations = 527040
        for _ in range(max_iterations):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)

        raise CronParseError(
            f"Could not find next run time for expression: {self.expression}"
        )

    def describe(self) -> str:
        """Return a human-readable description of the cron schedule.

        Handles complex field expressions (lists, ranges, steps) gracefully
        by falling back to the raw field text when parsing for display would
        be ambiguous.
        """
        parts = self.expression.strip().split()
        minute, hour, dom, month, dow = parts

        pieces: list[str] = []

        # Time description
        if minute != "*" and hour != "*":
            # Show raw values — handles steps/ranges cleanly
            pieces.append(f"at {hour}:{minute.zfill(2) if minute.isdigit() else minute}")
        elif hour != "*":
            pieces.append(f"during hour {hour}")
        elif minute != "*":
            pieces.append(f"at minute {minute}")
        else:
            pieces.append("every minute")

        # Day of week
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if dow != "*":
            try:
                # Attempt to render human-friendly names for simple cases
                dow_parts = dow.split(",")
                rendered: list[str] = []
                for part in dow_parts:
                    if "-" in part and "/" not in part:
                        start_s, end_s = part.split("-", 1)
                        rendered.append(f"{day_names[int(start_s)]}-{day_names[int(end_s)]}")
                    elif part.isdigit():
                        rendered.append(day_names[int(part)])
                    else:
                        # Steps or complex — use raw
                        rendered.append(part)
                pieces.append(", ".join(rendered))
            except (ValueError, IndexError):
                # Fall back to raw expression for anything unparseable
                pieces.append(f"dow {dow}")

        # Day of month
        if dom != "*":
            pieces.append(f"on day {dom}")

        # Month
        if month != "*":
            pieces.append(f"in month {month}")

        return " ".join(pieces)


def validate_cron(expression: str) -> str | None:
    """Validate a cron expression. Returns None if valid, error message if not."""
    try:
        CronExpression(expression)
        return None
    except CronParseError as e:
        return str(e)


def next_run_time(expression: str, after: Optional[datetime] = None) -> datetime:
    """Calculate the next run time for a cron expression."""
    cron = CronExpression(expression)
    return cron.next_run(after)
