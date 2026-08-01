"""Provider discovery, introspection, connection testing, and persistence.

This service bridges the webgui admin panel with the ``smart_albums``
``ProtocolsRegistry``. It exposes:

* The ordered list of configurable protocol *slots* (image, llm, embedding,
  cache) that the pipeline resolves at runtime.
* For each slot, the registered providers and an introspected parameter
  schema derived from each provider class ``__init__`` signature.
* A connection test that instantiates a provider with candidate parameters
  and invokes its ``health_check`` (for providers implementing
  ``IHealthCheck``).
* Persistence of the selected providers and their parameters to the
  ``settings`` table in the app database, so changes take effect without
  editing ``config.yaml``.

The parameter introspection reuses ``smart_albums.core.builder.get_init_schema``
so the webgui and CLI stay in lock-step about what each provider accepts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from smart_albums.cli.imports import import_clients
from smart_albums.core.builder import get_init_schema
from smart_albums.core.node import ConfigParam
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import (
    ICache,
    ICacheManager,
    IEmbeddingClient,
    IGeoClient,
    IHealthCheck,
    IImageClient,
    ILLMClient,
    IProgressReporter,
)

from webgui.config import ProviderConfig
from webgui.state import PROVIDERS_SETTINGS_KEY, state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol slot metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolSlot:
    """A configurable protocol slot resolved by the pipeline at runtime.

    Attributes:
        name: The config key under ``providers`` (e.g. ``"image"``). Derived
            from the protocol class name by stripping the ``I`` prefix and
            ``Client`` suffix, then lowercasing.
        protocol: The protocol type used to look providers up in the registry.
        label: Human-friendly title shown in the admin UI. Parsed from the
            protocol docstring before the colon.
        description: One-line explanation of the slot's role. Parsed from the
            protocol docstring after the colon.
    """

    name: str
    protocol: type
    label: str
    description: str


# Cross-cutting protocols that should NOT appear as configurable provider
# slots in the UI — they are infrastructure/internal concerns.
_EXCLUDED_PROTOCOLS: set[type] = {IHealthCheck, IProgressReporter, ICache}


def _derive_slot_name(protocol_cls: type) -> str:
    """Derive the slot name from a protocol class name.

    Rules:
    - Remove leading ``I`` prefix.
    - Remove trailing ``Client`` suffix.
    - Lowercase the result.

    Examples:
        IImageClient -> image
        ILLMClient -> llm
        IEmbeddingClient -> embedding
        ICacheManager -> cachemanager
        IGeoClient -> geo
    """
    name = protocol_cls.__name__
    # Strip leading 'I'
    if name.startswith("I") and len(name) > 1 and name[1].isupper():
        name = name[1:]
    # Strip trailing 'Client'
    if name.endswith("Client"):
        name = name[: -len("Client")]
    return name.lower()


def _parse_protocol_doc(protocol_cls: type) -> tuple[str, str]:
    """Parse label and description from protocol docstring.

    Expected format: ``\"\"\"Label:Description\"\"\"``

    Falls back to the class name as label and empty description if the
    docstring doesn't follow the convention.
    """
    doc = protocol_cls.__doc__ or ""
    # Take only the first line of the docstring
    first_line = doc.strip().split("\n")[0].strip()

    if ":" in first_line:
        label, _, description = first_line.partition(":")
        return label.strip(), description.strip()

    # Fallback: use class name as label
    return _derive_slot_name(protocol_cls).title(), first_line


def _build_protocol_slots() -> list[ProtocolSlot]:
    """Dynamically discover configurable protocol slots from the registry.

    Iterates all protocols registered in ProtocolsRegistry, excludes
    cross-cutting ones (IHealthCheck, IProgressReporter, ICache), and
    builds ProtocolSlot instances using docstring metadata.
    """
    _ensure_registered()

    slots: list[ProtocolSlot] = []
    for protocol_cls in ProtocolsRegistry.get_protocols():
        if protocol_cls in _EXCLUDED_PROTOCOLS:
            continue

        name = _derive_slot_name(protocol_cls)
        label, description = _parse_protocol_doc(protocol_cls)
        slots.append(
            ProtocolSlot(
                name=name,
                protocol=protocol_cls,
                label=label,
                description=description,
            )
        )

    # Sort alphabetically by name for stable ordering
    slots.sort(key=lambda s: s.name)
    return slots


def get_protocol_slots() -> list[ProtocolSlot]:
    """Return the list of configurable protocol slots.

    Dynamically discovers slots from the ProtocolsRegistry on first call,
    then caches the result for subsequent calls.
    """
    global _protocol_slots_cache
    if _protocol_slots_cache is None:
        _protocol_slots_cache = _build_protocol_slots()
    return _protocol_slots_cache


_protocol_slots_cache: list[ProtocolSlot] | None = None


# Backward-compatible alias — consumers can still import PROTOCOL_SLOTS,
# but it now delegates to dynamic discovery.
class _ProtocolSlotsProxy:
    """Lazy proxy that behaves like a list but builds slots on first access."""

    def __iter__(self):
        return iter(get_protocol_slots())

    def __len__(self):
        return len(get_protocol_slots())

    def __getitem__(self, index):
        return get_protocol_slots()[index]


PROTOCOL_SLOTS = _ProtocolSlotsProxy()


# ---------------------------------------------------------------------------
# Parameter / provider schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """A single provider constructor parameter, described for form rendering.

    Attributes:
        name: The constructor keyword argument name.
        kind: A widget hint — one of ``"str"``, ``"password"``, ``"int"``,
            ``"float"``, or ``"bool"``.
        required: Whether the parameter has no default (must be supplied).
        default: The default value, or ``None`` when required.
    """

    name: str
    kind: str
    required: bool
    default: Any


@dataclass(frozen=True)
class ProviderSpec:
    """A registered provider and its introspected configuration schema.

    Attributes:
        name: The provider's registry name (e.g. ``"immich"``).
        params: Ordered list of constructor parameters.
        supports_health_check: True if the provider implements ``IHealthCheck``
            and can therefore be connection-tested.
    """

    name: str
    params: list[ParamSpec]
    supports_health_check: bool


_SECRET_HINTS = ("api_key", "apikey", "password", "secret", "token")


def _classify_param(param: ConfigParam) -> str:
    """Map a ConfigParam to a UI widget kind.

    Because provider modules use ``from __future__ import annotations``, the
    ``type`` attribute is typically the annotation *string* (e.g. ``"str"``).
    We combine the annotation text with the default value's runtime type to
    pick the most appropriate widget.
    """
    annotation = param.type
    type_name = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    ).lower()

    default = param.default
    key = param.key.lower()

    # bool must be checked before int (bool is a subclass of int).
    if "bool" in type_name or isinstance(default, bool):
        return "bool"

    if any(hint in key for hint in _SECRET_HINTS):
        return "password"

    if "float" in type_name or isinstance(default, float):
        return "float"

    if "int" in type_name or (
        isinstance(default, int) and not isinstance(default, bool)
    ):
        return "int"

    return "str"


def _param_spec_from_config(param: ConfigParam) -> ParamSpec:
    """Convert a core ConfigParam into a webgui ParamSpec."""
    return ParamSpec(
        name=param.key,
        kind=_classify_param(param),
        required=param.required,
        default=param.default,
    )


_registered = False


def _ensure_registered() -> None:
    """Trigger client registration exactly once.

    ``import_clients`` is idempotent (module imports are cached), but we guard
    with a flag to avoid repeated import churn on every UI interaction.
    """
    global _registered
    if not _registered:
        import_clients()
        _registered = True


def _health_check_classes() -> set[type]:
    """Return the set of provider classes registered under IHealthCheck."""
    return set(ProtocolsRegistry.get_providers(IHealthCheck).values())


def get_providers_for(slot: ProtocolSlot) -> list[ProviderSpec]:
    """Return the providers registered for a slot, with introspected schemas.

    Args:
        slot: The protocol slot to enumerate providers for.

    Returns:
        Providers sorted by name, each with its parameter schema and a flag
        indicating whether it can be connection-tested.
    """
    _ensure_registered()
    hc_classes = _health_check_classes()

    specs: list[ProviderSpec] = []
    for name, cls in sorted(ProtocolsRegistry.get_providers(slot.protocol).items()):
        schema = get_init_schema(cls)
        params = [_param_spec_from_config(cp) for cp in schema.values()]
        specs.append(
            ProviderSpec(
                name=name,
                params=params,
                supports_health_check=cls in hc_classes,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Connection testing
# ---------------------------------------------------------------------------


async def test_connection(
    slot: ProtocolSlot,
    provider_name: str,
    params: dict[str, Any],
    timeout: float = 60.0,
) -> tuple[bool, str]:
    """Instantiate a provider and run its health check.

    Args:
        slot: The protocol slot the provider belongs to.
        provider_name: The registry name of the provider to test.
        params: Candidate constructor parameters (as entered in the form).
        timeout: Maximum seconds to wait for the health check.

    Returns:
        A ``(ok, message)`` tuple. ``ok`` is True when the backing service is
        reachable; ``message`` is a short human-readable status.
    """
    _ensure_registered()

    if not provider_name:
        return False, "No provider selected"

    try:
        client = ProtocolsRegistry.create(slot.protocol, provider_name, params)
    except Exception as exc:  # noqa: BLE001 — surface any construction error
        logger.warning("Provider instantiation failed: %s", exc)
        return False, f"Could not create provider: {exc}"

    if not isinstance(client, IHealthCheck):
        return False, "This provider does not support connection testing"

    async def _run() -> bool:
        # Enter the async context manager when available so clients that load
        # resources on entry (e.g. local models) are exercised realistically.
        if hasattr(client, "__aenter__"):
            async with client:  # type: ignore[attr-defined]
                return await client.health_check()
        return await client.health_check()

    try:
        ok = await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        return False, f"Timed out after {int(timeout)}s"
    except Exception as exc:  # noqa: BLE001 — health checks must not crash the UI
        logger.warning("Connection test error: %s", exc)
        return False, f"Error: {exc}"

    url = getattr(client, "health_check_url", "")
    if ok:
        return True, f"Connected{f' — {url}' if url else ''}"
    return False, f"Not reachable{f' — {url}' if url else ''}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def save(providers: dict[str, ProviderConfig]) -> None:
    """Persist provider configuration to the database and reload live config.

    Writes the full provider mapping to the ``settings`` table under the
    ``providers`` key, then updates ``state.config.providers`` in place so
    the change is visible immediately without a process restart.

    Args:
        providers: Mapping of slot name -> ProviderConfig to save.
    """
    raw = {name: {"selected": pc.selected, "params": pc.params} for name, pc in providers.items()}
    await state.db.set_setting(PROVIDERS_SETTINGS_KEY, raw)
    await state.reload_providers()
