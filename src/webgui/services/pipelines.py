"""Pipeline schema introspection for the webgui.

Provides a cached list of registered pipelines and their serialized config
schemas. The schemas drive the dynamic form renderer on the new-job page.
"""

from __future__ import annotations

import logging
from typing import Any

from smart_albums.cli.imports import import_all_stages
from smart_albums.core.builder import get_pipeline_config_schema
from smart_albums.core.node import ConfigParam, ResolvedComposite
from smart_albums.pipelines.registry import get_pipeline, get_pipelines

logger = logging.getLogger(__name__)

_schemas: dict[str, dict[str, Any]] = {}


def _type_to_str(t: Any) -> str:
    """Convert a type annotation to a JSON/UI-friendly string."""
    if isinstance(t, type):
        return t.__name__
    return str(t)


def _serialize_param(param: ConfigParam) -> dict[str, Any]:
    """Serialize a ConfigParam to a dict for the form renderer."""
    return {
        "key": param.key,
        "type": _type_to_str(param.type),
        "default": param.default,
        "required": param.required,
        "description": param.description or None,
        "min": param.min,
        "max": param.max,
        "choices": param.choices,
        "widget": param.widget,
    }


def _serialize_schema(schema: list[Any]) -> list[dict[str, Any]]:
    """Recursively serialize a PipelineConfigSchema into a JSON tree."""
    result: list[dict[str, Any]] = []

    for item in schema:
        name_tuple, params_dict = item
        stage_name, alias_ = name_tuple

        step: dict[str, Any] = {
            "stage": stage_name,
            "alias": alias_,
            "params": {},
            "children": None,
        }

        for key, param in params_dict.items():
            if isinstance(param, ConfigParam):
                step["params"][key] = _serialize_param(param)
            elif isinstance(param, ResolvedComposite):
                step["children"] = {
                    "key": param.key,
                    "description": param.description or None,
                    "branches": [
                        {
                            "name": branch_name,
                            "steps": _serialize_schema(branch_schema),
                        }
                        for branch_name, branch_schema in param.children
                    ],
                }

        result.append(step)

    return result


def get_all_schemas() -> dict[str, dict[str, Any]]:
    """Return cached pipeline schemas (label, description, schema tree).

    Triggers stage/pipeline registration on first call. Subsequent calls
    return the cached result.

    Includes both built-in registered pipelines and user-defined pipelines
    from the database (prefixed with "user:").
    """
    global _schemas
    if _schemas:
        return _schemas

    import_all_stages()
    import smart_albums.pipelines  # noqa: F401 — triggers @register_pipeline

    for name in get_pipelines():
        info = get_pipeline(name)
        if info is None:
            continue
        _schemas[name] = {
            "name": name,
            "label": info.label,
            "description": info.description,
            "schema": _serialize_schema(get_pipeline_config_schema(info.pipeline)),
        }

    return _schemas


async def get_all_schemas_with_user_pipelines() -> dict[str, dict[str, Any]]:
    """Return both built-in and user-defined pipeline schemas.

    User pipelines are prefixed with "user:" and include a full config
    schema generated from their step definitions, so they can be
    configured and have presets just like built-in pipelines.
    """
    from smart_albums.core.spec import alias as make_alias
    from smart_albums.core.registry import has_stage, get_stage
    from smart_albums.core.node import CompositeParam

    from webgui.services.user_pipelines import USER_PIPELINE_PREFIX
    from webgui.state import state

    import_all_stages()

    # Start with built-in schemas
    schemas = dict(get_all_schemas())

    def _reconstruct_pipeline(steps: list[Any]) -> list[Any]:
        """Recursively reconstruct a Pipeline list from stored JSON steps.

        Fork stages are instantiated as real PipelineNode objects so that
        ``get_pipeline_config_schema`` can resolve their CompositeParam and
        produce nested schemas for the form renderer.
        """
        pipeline_steps: list[Any] = []
        for step in steps:
            if not isinstance(step, (list, tuple)) or len(step) < 2:
                continue
            stage_name = step[0]
            config = step[1] if len(step) > 1 else {}
            alias_name = step[2] if len(step) > 2 else None

            if not has_stage(stage_name):
                continue

            # For fork stages, build a real node instance so CompositeParam resolves
            if stage_name in ("fork", "fork.by_selection") and isinstance(config, dict) and "branches" in config:
                branches_raw = config["branches"]
                # Recursively reconstruct each branch's pipeline
                reconstructed_branches: list[Any] = []
                if isinstance(branches_raw, list):
                    for branch in branches_raw:
                        if isinstance(branch, (list, tuple)) and len(branch) == 2:
                            branch_name, branch_steps = branch
                            reconstructed_branches.append(
                                [branch_name, _reconstruct_pipeline(branch_steps)]
                            )
                elif isinstance(branches_raw, dict):
                    for branch_name, branch_steps in branches_raw.items():
                        reconstructed_branches.append(
                            [branch_name, _reconstruct_pipeline(branch_steps)]
                        )

                # Build the fork node instance using its CompositeParam.build
                cls = get_stage(stage_name)
                if cls is not None:
                    composite_params = [
                        p for p in cls._config_schema if isinstance(p, CompositeParam)
                    ]
                    if composite_params and composite_params[0].build is not None:
                        try:
                            kwargs = composite_params[0].build(reconstructed_branches)
                            instance = cls(**kwargs)
                            # Use the instance directly as a pipeline step
                            if alias_name:
                                pipeline_steps.append(make_alias(alias_name, instance))
                            else:
                                pipeline_steps.append(instance)
                            continue
                        except Exception:
                            logger.debug(
                                "Failed to build fork instance for schema, falling back",
                                exc_info=True,
                            )

                # Fallback: plain tuple (no instance, branches won't show in schema)
                raw_step: Any = (stage_name, {"branches": reconstructed_branches})
                if alias_name:
                    pipeline_steps.append(make_alias(alias_name, raw_step))
                else:
                    pipeline_steps.append(raw_step)
            else:
                raw_step = (stage_name, config) if config else stage_name
                if alias_name:
                    pipeline_steps.append(make_alias(alias_name, raw_step))
                else:
                    pipeline_steps.append(raw_step)
        return pipeline_steps

    # Add user-defined pipelines with full schema introspection
    user_pipelines = await state.db.list_user_pipelines()
    for up in user_pipelines:
        key = f"{USER_PIPELINE_PREFIX}{up['id']}"

        # Reconstruct a Pipeline list from stored steps
        pipeline_steps = _reconstruct_pipeline(up.get("steps", []))

        # Generate schema from the reconstructed pipeline
        schema_tree: list[dict[str, Any]] = []
        if pipeline_steps:
            try:
                raw_schema = get_pipeline_config_schema(pipeline_steps)
                schema_tree = _serialize_schema(raw_schema)
            except Exception:
                logger.warning(
                    "Failed to generate schema for user pipeline %s",
                    up["id"],
                    exc_info=True,
                )

        schemas[key] = {
            "name": key,
            "label": f"📋 {up['name']}",
            "description": up.get("description", ""),
            "schema": schema_tree,
            "is_user_pipeline": True,
        }

    return schemas


def invalidate_cache() -> None:
    """Clear the cached schemas so they are rebuilt on next access."""
    global _schemas
    _schemas = {}
