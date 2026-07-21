"""PipelineContext — the mutable state bag that flows through all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from smart_albums.core.models import Asset
from smart_albums.core.protocols import (
    IEmbeddingClient,
    IImageClient,
    ILLMClient,
    IProgressReporter,
    ICacheManager
)


@dataclass
class PipelineContext:
    """The single mutable state bag that flows through all stages.

    Stages read configuration from `config`, interact with external services
    via `image_client`, `llm_client`, and `embedding_client`, report progress
    through `progress`, and mutate the `assets` list. The `stats` dict
    accumulates pipeline-level metrics (timing, counts, etc.).

    Partition tracking fields (`partition_name`, `partition_id`,
    `parent_partition_id`, `partition_depth`) record where this context sits
    in a nested partition tree. A depth of 0 means "root / not partitioned".
    """

    config: dict[str, Any] = field(default_factory=dict)
    image_client: IImageClient | None = None
    llm_client: ILLMClient | None = None
    embedding_client: IEmbeddingClient | None = None
    progress: IProgressReporter | None = None
    cache_manager: ICacheManager | None = None

    assets: list[Asset] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Partition tracking
    partition_name: str = ""
    partition_id: str = ""
    parent_partition_id: Optional[str] = None
    partition_depth: int = 0

ContextBatch = list[PipelineContext]
