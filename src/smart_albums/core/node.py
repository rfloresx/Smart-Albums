from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from smart_albums.core.context import PipelineContext, ContextBatch

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ConfigParam:
    """Describes a single configuration key consumed by a stage.

    The optional ``widget`` dict provides rendering hints to UI frontends.
    Example widgets:
        {"type": "file", "glob": "*.md"}   — file picker filtered by glob
        {"type": "textarea", "rows": 12}   — multiline text input
        {"type": "color"}                  — color picker
    """
    key: str
    type: Any
    default: Any = None
    required: bool = False
    description: str = ""
    min: float | None = None
    max: float | None = None
    choices: list[str] | None = None
    widget: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompositeParam:
    """Describes a config key that holds one or more child pipelines.

    The ``resolve`` callable receives the node instance and returns
    a list of (name, Pipeline) pairs representing the child pipelines.

    The ``build`` callable receives the raw config value (e.g. a dict of
    {branch_name: pipeline_list} from JSON) and returns kwargs to pass
    to the node constructor.
    """
    key: str
    description: str = ""
    resolve: Any = None  # Callable[[PipelineNode], list[tuple[str, Pipeline]]]
    build: Any = None    # Callable[[Any], dict[str, Any]] — raw config → constructor kwargs


@dataclass
class ResolvedComposite:
    """A resolved CompositeParam — holds recursive pipeline config schemas.

    Produced by ``get_pipeline_config_schema`` when a CompositeParam is
    resolved against a live node instance.
    """
    key: str
    description: str = ""
    children: list[Any] = field(default_factory=list)  # list[tuple[str, PipelineConfigSchema]]


class PipelineNode(ABC):
    _stage_name: ClassVar[str]
    _config_schema: ClassVar[tuple[ConfigParam | CompositeParam, ...]] = ()
    _description : str = ""

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = {}
        self.config = {
            param.key: param.default
            for param in self._config_schema
            if isinstance(param, ConfigParam)
        }
        self.config.update(config)

    @property
    def name(self) -> str:
        return self._stage_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> tuple[ConfigParam | CompositeParam, ...]:
        return self._config_schema

    def get(self, key: str) -> Any:
        return self.config.get(key)

    @abstractmethod
    async def func(self, contexts: ContextBatch) -> ContextBatch:
        ...

class Stage(PipelineNode, ABC):
    """Base class for class-based stages."""

    async def func(self, contexts: ContextBatch) -> ContextBatch:
        results : ContextBatch = []
        for ctx in contexts:
            results.extend(await self.run(ctx))
        return results

    @abstractmethod
    async def run(self, context: PipelineContext) -> ContextBatch:
        ...


class CompositeNode(PipelineNode):
    """A node composed of an internal sequence of sub-nodes.

    Subclasses define ``_sub_pipeline`` — a standard ``Pipeline`` list using
    ``alias()`` to tag each step. The composite exposes the union of all child
    config schemas as its own, routing config values to the appropriate child
    at execution time.

    Config key routing rules:
      - If a key is unique across all children, it's exposed flat (no prefix).
      - If multiple children share the same key, it's namespaced as
        ``<alias>.<key>`` to disambiguate.

    Example::

        @stage("dedup.phash_composite")
        class DedupPHashComposite(CompositeNode):
            _sub_pipeline = [
                alias("analyze", AnalyzePHash),
                alias("partition", PartitionPHash),
                alias("select", SelectBest),
                alias("merge", MergeConcat),
            ]
    """

    _sub_pipeline: ClassVar[list[Any]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-generate _config_schema from the sub-pipeline when the class is defined."""
        super().__init_subclass__(**kwargs)
        if cls._sub_pipeline:
            cls._config_schema = cls._build_config_schema()

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._resolved_pipeline: list[Any] = self._apply_config()

    def _apply_config(self) -> list[Any]:
        """Route config values to child step overrides using the alias map."""
        from smart_albums.core.builder import resolve_overrides, build_alias_map

        if not self.config:
            return list(self._sub_pipeline)

        # Build the key→alias routing table
        key_to_alias = self._build_key_routing()

        overrides: dict[str, dict[str, Any]] = {}
        for key, value in self.config.items():
            if value is None:
                continue
            if "." in key:
                # Already namespaced: "partition.threshold" → alias="partition", param="threshold"
                alias_part, _, param = key.rpartition(".")
                # Verify the alias exists
                alias_map = build_alias_map(self._sub_pipeline)
                if alias_part in alias_map:
                    overrides.setdefault(alias_part, {})[param] = value
            else:
                # Flat key — route via the lookup table
                target_alias = key_to_alias.get(key)
                if target_alias is not None:
                    overrides.setdefault(target_alias, {})[key] = value

        if not overrides:
            return list(self._sub_pipeline)

        return resolve_overrides(self._sub_pipeline, overrides)

    def _build_key_routing(self) -> dict[str, str]:
        """Build a map from flat config key → child alias that owns it.

        Only maps keys that are unique across all children (non-ambiguous).
        """
        from smart_albums.core.builder import build_alias_map, _extract_stage_name
        import smart_albums.core.registry as registry

        alias_map = build_alias_map(self._sub_pipeline)
        # Track which alias owns each key and how many aliases declare it
        key_owners: dict[str, list[str]] = defaultdict(list)

        for alias_name, idx in alias_map.items():
            step = self._sub_pipeline[idx]
            stage_name = _extract_stage_name(step)
            params = registry.get_stage_config(stage_name)
            for param in params:
                if isinstance(param, ConfigParam):
                    key_owners[param.key].append(alias_name)

        # Only map keys owned by exactly one child
        return {
            key: owners[0]
            for key, owners in key_owners.items()
            if len(owners) == 1
        }

    @classmethod
    def _build_config_schema(cls) -> tuple[ConfigParam, ...]:
        """Derive the composite config schema from child node schemas."""
        from smart_albums.core.builder import build_alias_map, _extract_stage_name
        import smart_albums.core.registry as registry

        alias_map = build_alias_map(cls._sub_pipeline)

        # Count how many children declare each key
        key_count: dict[str, int] = defaultdict(int)
        all_params: list[tuple[str, ConfigParam]] = []

        for alias_name, idx in alias_map.items():
            step = cls._sub_pipeline[idx]
            stage_name = _extract_stage_name(step)
            params = registry.get_stage_config(stage_name)
            for param in params:
                if isinstance(param, ConfigParam):
                    key_count[param.key] += 1
                    all_params.append((alias_name, param))

        schema: list[ConfigParam] = []
        for alias_name, param in all_params:
            if key_count[param.key] > 1:
                # Namespace to avoid collision
                key = f"{alias_name}.{param.key}"
            else:
                key = param.key
            schema.append(ConfigParam(
                key=key,
                type=param.type,
                default=param.default,
                required=param.required,
                description=param.description,
                min=param.min,
                max=param.max,
                choices=param.choices,
                widget=param.widget,
            ))

        return tuple(schema)

    async def func(self, contexts: ContextBatch) -> ContextBatch:
        """Execute the internal sub-pipeline."""
        from smart_albums.core.runner import run_pipeline
        return await run_pipeline(self._resolved_pipeline, contexts)

