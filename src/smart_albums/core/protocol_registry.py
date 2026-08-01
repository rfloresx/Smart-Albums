"""Protocol-based service registry for dependency injection."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class ProtocolsRegistry:
    _registry: dict[type, dict[str, type]] = {}
    _instances: dict[type, Any] = {}

    @staticmethod
    def register(name: str, protocol: type) -> Any:
        def decorator(cls: type) -> type:
            if protocol not in ProtocolsRegistry._registry:
                ProtocolsRegistry._registry[protocol] = {}
            ProtocolsRegistry._registry[protocol][name] = cls
            return cls
        return decorator

    @staticmethod
    def get_registry() -> dict[type, dict[str, type]]:
        return ProtocolsRegistry._registry

    @staticmethod
    def get_protocols() -> list[type]:
        return list(ProtocolsRegistry._registry.keys())

    @staticmethod
    def get_providers(protocol: type) -> dict[str, type]:
        return ProtocolsRegistry._registry.get(protocol, {})

    @staticmethod
    def create(protocol: type, name: str, config: dict[str, Any]) -> Any:
        """Create an instance and store it in the internal instances map."""
        if protocol not in ProtocolsRegistry._registry:
            raise ValueError(f"No clients registered for protocol {protocol}")
        if name not in ProtocolsRegistry._registry[protocol]:
            raise ValueError(f"No client named {name} registered for protocol {protocol}")
        instance = ProtocolsRegistry._registry[protocol][name](**config)
        ProtocolsRegistry._instances[protocol] = instance
        return instance

    @staticmethod
    def get_instance(protocol: type[T]) -> T | None:
        """Return the live instance for a protocol, or None if not created."""
        return ProtocolsRegistry._instances.get(protocol)

    @staticmethod
    def set_instance(protocol: type, instance: Any) -> None:
        """Manually register a pre-built instance (useful for testing / webgui)."""
        ProtocolsRegistry._instances[protocol] = instance

    @staticmethod
    def clear_instances() -> None:
        """Remove all live instances. Called between pipeline runs or in tests."""
        ProtocolsRegistry._instances.clear()
