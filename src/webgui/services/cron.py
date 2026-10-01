"""Cron expression parser and next-run calculator.

Supports standard 5-field cron expressions:
    minute hour day_of_month month day_of_week

Field values:
    * — any value
    N — exact value
    N-M — range
    N/S — step (every S starting at N; */S is shorthand for 0/S)
    N,M,... — list

Day of week: 0=Sunday .. 6=Saturday (standard/Vixie cron convention), and 7
is also accepted as an alias for Sunday. When BOTH day_of_month and
day_of_week are restricted (neither is the literal "*"), a day matches if
EITHER field matches (OR), per standard cron semantics — e.g. ``0 0 1 * 1``
means "midnight on the 1st of the month, OR any Monday", not "the 1st only
if it happens to be a Monday". When only one of the two fields is
restricted, that field alone determines the match (the "*" field imposes no
constraint), which is equivalent to AND in that case.

All schedules run in UTC. There is no per-schedule timezone support.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# How many years ahead next_run() will search before giving up. Large enough
# to cover the widest realistic gap between Feb-29 occurrences (normally 4
# years, up to 8 around a non-leap century boundary), while keeping the
# day-matching loop (the only unbounded-feeling part) cheap: even in the
# worst case (an expression that can never match, e.g. day 30 of February),
# this only means "check ~8*12*31 candidate days", not "check every minute
# for 8 years" — see next_run()'s docstring for why this stays fast.
_MAX_YEARS_AHEAD = 8


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

        # Day-of-week accepts 0-7; 7 is a standard alias for Sunday (0).
        raw_dow = _parse_field(parts[4], 0, 7)
        self.days_of_week = {0 if v == 7 else v for v in raw_dow}

        # Whether each day field was the literal wildcard "*" — this drives
        # the AND/OR combination rule in _day_matches, and must be recorded
        # from the raw text (not inferred from the parsed value set), since
        # e.g. "0-6" for day_of_week produces the same set as "*" but is
        # written as a restriction and should still combine with AND when
        # day_of_month is also restricted... actually per Vixie cron, the
        # OR rule is textual: it only applies when the field is not the
        # literal "*" character, regardless of what set it parses to.
        self._dom_is_star = parts[2].strip() == "*"
        self._dow_is_star = parts[4].strip() == "*"

    def _day_matches(self, year: int, month: int, day: int) -> bool:
        """Check whether a given calendar day matches the day fields.

        Applies the standard cron day-of-month / day-of-week combination
        rule: OR when both fields are restricted, otherwise whichever field
        is restricted (the "*" field always matches).
        """
        if self._dom_is_star and self._dow_is_star:
            return True

        dom_ok = day in self.days_of_month

        # Python's date.weekday() is 0=Monday..6=Sunday. Convert to the
        # standard cron convention (0=Sunday..6=Saturday) used by
        # self.days_of_week.
        py_weekday = date(year, month, day).weekday()
        cron_dow = (py_weekday + 1) % 7
        dow_ok = cron_dow in self.days_of_week

        if self._dom_is_star:
            return dow_ok
        if self._dow_is_star:
            return dom_ok
        return dom_ok or dow_ok

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.month in self.months
            and self._day_matches(dt.year, dt.month, dt.day)
        )

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next datetime matching this expression after *after*.

        Searches field-by-field (year → month → day → hour → minute)
        instead of stepping minute by minute. Day-of-month/day-of-week
        matching is checked before any hour/minute work, so even an
        expression that can never match (e.g. "day 30 of February") only
        costs a few thousand cheap date checks — roughly
        ``_MAX_YEARS_AHEAD * 12 * 31`` — rather than scanning up to
        ``_MAX_YEARS_AHEAD`` years of individual minutes. This also means a
        legitimately rare expression like "Feb 29 at midnight" is found
        correctly even when the next occurrence is a few years out, instead
        of raising because a fixed one-year lookahead ran out.

        Raises:
            CronParseError: If no matching time is found within
                ``_MAX_YEARS_AHEAD`` years (the expression can never match,
                e.g. day-of-month 30 combined with month=February only).
        """
        if after is None:
            after = datetime.now(timezone.utc)

        start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        for year_offset in range(_MAX_YEARS_AHEAD + 1):
            year = start.year + year_offset
            for month in range(1, 13):
                if month not in self.months:
                    continue
                if year == start.year and month < start.month:
                    continue

                days_in_month = calendar.monthrange(year, month)[1]
                first_day = 1
                if year == start.year and month == start.month:
                    first_day = start.day

                for day in range(first_day, days_in_month + 1):
                    if not self._day_matches(year, month, day):
                        continue

                    is_start_day = (
                        year == start.year
                        and month == start.month
                        and day == start.day
                    )

                    for hour in sorted(self.hours):
                        if is_start_day and hour < start.hour:
                            continue
                        for minute in sorted(self.minutes):
                            if (
                                is_start_day
                                and hour == start.hour
                                and minute < start.minute
                            ):
                                continue
                            return datetime(
                                year, month, day, hour, minute,
                                tzinfo=start.tzinfo,
                            )

        raise CronParseError(
            f"Could not find a matching run time within {_MAX_YEARS_AHEAD} "
            f"years for expression: {self.expression}"
        )

    def describe(self) -> str:
        """Return a human-readable description of the cron schedule.

        Handles complex field expressions (lists, ranges, steps) gracefully
        by falling back to the raw field text when parsing for display would
        be ambiguous. All times are UTC.
        """
        parts = self.expression.strip().split()
        minute, hour, dom, month, dow = parts

        pieces: list[str] = []

        # Time description
        if minute != "*" and hour != "*":
            # Show raw values — handles steps/ranges cleanly
            pieces.append(f"at {hour}:{minute.zfill(2) if minute.isdigit() else minute} UTC")
        elif hour != "*":
            pieces.append(f"during hour {hour} UTC")
        elif minute != "*":
            pieces.append(f"at minute {minute}")
        else:
            pieces.append("every minute")

        # Day of week — index 0=Sunday..6=Saturday, matching the standard
        # cron convention used for parsing. "7" (also Sunday) is mapped to
        # index 0 via modulo so it renders instead of raising IndexError.
        day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        if dow != "*":
            try:
                # Attempt to render human-friendly names for simple cases
                dow_parts = dow.split(",")
                rendered: list[str] = []
                for part in dow_parts:
                    if "-" in part and "/" not in part:
                        start_s, end_s = part.split("-", 1)
                        rendered.append(
                            f"{day_names[int(start_s) % 7]}-{day_names[int(end_s) % 7]}"
                        )
                    elif part.isdigit():
                        rendered.append(day_names[int(part) % 7])
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
    """Validate a cron expression.

    Returns None if valid, or an error message if not. Validation also
    confirms a next run time can actually be computed (catching expressions
    that parse but can never match, e.g. day-of-month 30 restricted to
    February), not just that the field syntax is well-formed.
    """
    try:
        cron = CronExpression(expression)
        cron.next_run()
        return None
    except CronParseError as e:
        return str(e)


def next_run_time(expression: str, after: Optional[datetime] = None) -> datetime:
    """Calculate the next run time for a cron expression."""
    cron = CronExpression(expression)
    return cron.next_run(after)
