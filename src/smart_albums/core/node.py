from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from smart_albums.core.context import PipelineContext, ContextBatch

if TYPE_CHECKING:
    from smart_albums.core.spec import Pipeline


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

