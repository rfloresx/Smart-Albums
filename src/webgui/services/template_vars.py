"""Shared template-variable resolution for pipeline_settings.

Pipeline settings dicts can contain ``{VAR_NAME}`` placeholders (defined
either in the New Job page's inline editor or carried over from a saved
preset). This module provides the single resolution implementation used by
every code path that can launch a job with such settings: the New Job page,
schedule firing, and "Run Now".

Previously, only ``pages/new_job.py``'s submit handler called this logic.
Schedules (both the periodic scheduler loop and the "Run Now" button) sent
``pipeline_settings`` straight to ``jobs_service.create_job`` with any
``{VAR}`` placeholders still literally in them, since presets store their
raw (unresolved) settings and only ``new_job.py`` ever substituted values
in (WG-19).
"""

from __future__ import annotations

import re
from typing import Any

# Pattern matching a single {VAR_NAME} placeholder.
TEMPLATE_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_template_variables(
    pipeline_settings: dict[str, Any],
    variables: dict[str, str] | None,
) -> dict[str, Any]:
    """Replace ``{VAR}`` placeholders in ``pipeline_settings`` string values.

    Only replaces variables that are defined in ``variables``. Unresolved
    placeholders are left as-is (callers that want to reject unresolved
    references should check with ``find_undefined_variables`` first).

    Note: only resolves one level deep (alias -> flat key/value dict).
    Nested dict values are not recursed into — this matches the shape
    ``pipeline_settings`` actually has (a dict of alias -> {param: value}).
    """
    if not variables:
        return pipeline_settings

    def _substitute(value: Any) -> Any:
        if isinstance(value, str):
            def _replace(m: "re.Match[str]") -> str:
                var_name = m.group(1)
                return variables.get(var_name, m.group(0))
            return TEMPLATE_PATTERN.sub(_replace, value)
        return value

    resolved: dict[str, Any] = {}
    for alias, settings in pipeline_settings.items():
        if isinstance(settings, dict):
            resolved[alias] = {k: _substitute(v) for k, v in settings.items()}
        else:
            resolved[alias] = _substitute(settings)
    return resolved


def find_undefined_variables(
    pipeline_settings: dict[str, Any],
    variables: dict[str, str],
) -> set[str]:
    """Return the set of {VAR} names used in settings but not in ``variables``."""
    defined = set(variables.keys())
    used: set[str] = set()

    for _alias, settings in pipeline_settings.items():
        if isinstance(settings, dict):
            for v in settings.values():
                if isinstance(v, str):
                    used.update(TEMPLATE_PATTERN.findall(v))
        elif isinstance(settings, str):
            used.update(TEMPLATE_PATTERN.findall(settings))

    return used - defined
