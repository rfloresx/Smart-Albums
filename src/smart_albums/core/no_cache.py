"""No-op cache manager provider."""

from __future__ import annotations

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import ICacheManager


@ProtocolsRegistry.register("disabled", ICacheManager)
class NoCacheManager:
    """No-op cache manager that always returns None from the factory.

    Registered as the "disabled" provider so that pipeline execution can
    proceed without caching when no cache_dir is configured.
    """

    def get_cache(self, cache_name: str) -> None:
        """Return None — signals to stages that caching is disabled."""
        return None
