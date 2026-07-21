"""Pipeline specification types — defines the shape of pipeline definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from smart_albums.core.node import PipelineNode


@dataclass(frozen=True)
class AliasedStep:
    """A pipeline step tagged with a human-readable alias for config resolution.

    The alias identifies the step's *role* in the pipeline (e.g. "duplicate",
    "scenes") independently of the underlying node type name.
    """
    alias: str
    step: Any  # StepInput at runtime — avoids circular import issues


def alias(name: str, step: Any) -> AliasedStep:
    """Tag a pipeline step with a config alias.

    Args:
        name: Human-readable alias for this step (used as config key).
        step: The underlying pipeline step (class, string, tuple, instance).

    Returns:
        An AliasedStep wrapping the step with the given alias.
    """
    return AliasedStep(alias=name, step=step)


@dataclass(frozen=True)
class StepSpec:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    instance: "PipelineNode | None" = field(default=None, repr=False)
    alias: str | None = field(default=None)

    children: list["PipelineSpec"] = field(
        default_factory=list
    )


# At runtime PipelineNode isn't imported, so we use Any for the instance slot.
# Type checkers see the proper Union via TYPE_CHECKING.
if TYPE_CHECKING:
    StepInput = Union[
        str,
        type["PipelineNode"],
        "PipelineNode",
        StepSpec,
        tuple[str, dict[str, Any]],
        AliasedStep,
    ]
else:
    StepInput = Any

Pipeline = list[StepInput]
PipelineSpec = list[StepSpec]
