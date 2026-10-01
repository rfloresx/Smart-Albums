"""Shared helpers for the LLM vision clients (ollama, llamacpp, openai_llm).

Centralizes two things that were previously copy-pasted (and subtly wrong)
in each client:

* :func:`validate_against_schema` — a stricter JSON-schema validator
  (CL-07). The old per-client copies didn't check the result was a dict,
  accepted ``bool`` as a number, ignored ``minimum``/``maximum``/``enum``/
  ``integer``, and let NaN through.
* :func:`prepare_image_for_vision` — normalizes arbitrary image bytes to a
  format the vision APIs actually accept (CL-08). The old per-client code
  only converted WebP, mislabeled everything else as ``image/jpeg`` while
  sending the original (HEIC/TIFF/BMP/GIF) bytes, never registered
  ``pillow-heif`` (so HEIC couldn't decode at all), and on conversion
  failure logged a warning and sent the bad bytes anyway.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Try to register HEIC/HEIF support so iPhone images can be decoded (CL-08).
# pillow-heif is an optional dependency; if it isn't installed, HEIC input
# simply can't be converted and prepare_image_for_vision reports an error
# rather than silently sending undecodable bytes.
try:  # pragma: no cover - depends on optional dependency being installed
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except Exception:  # pragma: no cover
    _HEIF_AVAILABLE = False

# Formats the vision endpoints accept directly without re-encoding.
_PASSTHROUGH_MIME = ("image/jpeg", "image/png")

# Target max pixels when downscaling (~2 megapixels). Large images waste
# tokens/VRAM and some endpoints reject them outright.
_MAX_PIXELS = 2_000_000


class ImageConversionError(RuntimeError):
    """Raised when image bytes can't be decoded/converted for a vision API."""


def detect_mime(image_bytes: bytes) -> str:
    """Best-effort MIME detection from magic bytes.

    Returns one of image/png, image/webp, image/jpeg, image/gif,
    image/bmp, image/tiff, or "application/octet-stream" when unknown.
    """
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:2] == b"BM":
        return "image/bmp"
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    # HEIC/HEIF: "ftyp" box with a heic/heif/mif1 brand near the start.
    if image_bytes[4:8] == b"ftyp" and image_bytes[8:12] in (
        b"heic", b"heix", b"hevc", b"heif", b"mif1", b"msf1",
    ):
        return "image/heic"
    return "application/octet-stream"


def prepare_image_for_vision(image_bytes: bytes) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` safe to send to a vision API.

    JPEG/PNG are passed through unchanged. Everything else (WebP, HEIC,
    TIFF, BMP, GIF, or unknown) is decoded via Pillow and re-encoded to
    JPEG, and the result is downscaled to roughly 2 MP. Unlike the old
    per-client code, a decode/convert failure raises
    :class:`ImageConversionError` instead of returning the original bytes
    with a bogus ``image/jpeg`` label (CL-08) — the caller turns that into
    a proper per-image error rather than sending garbage to the model.
    """
    mime = detect_mime(image_bytes)
    if mime in _PASSTHROUGH_MIME:
        return image_bytes, mime

    if mime == "image/heic" and not _HEIF_AVAILABLE:
        raise ImageConversionError(
            "HEIC image received but pillow-heif is not installed, so it "
            "cannot be converted for the vision API. Install the 'heif' "
            "extra (pillow-heif) or supply JPEG/PNG."
        )

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Downscale large images to ~2 MP to bound token/VRAM cost.
        pixels = img.width * img.height
        if pixels > _MAX_PIXELS:
            scale = math.sqrt(_MAX_PIXELS / pixels)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except ImageConversionError:
        raise
    except Exception as exc:
        raise ImageConversionError(
            f"Failed to convert {mime} image to JPEG for the vision API: {exc}"
        ) from exc


def validate_against_schema(data: Any, schema: dict[str, Any]) -> str | None:
    """Validate ``data`` against a simplified JSON schema.

    Returns ``None`` if valid, else an error message. Stricter than the old
    per-client copies (CL-07):

    - verifies ``data`` is actually a dict (``"x" in 5`` used to raise
      TypeError);
    - treats ``bool`` as neither number nor integer;
    - enforces ``integer`` type, finite ``number`` (rejects NaN/inf), and
      ``minimum``/``maximum``/``enum`` constraints.
    """
    if not isinstance(data, dict):
        return f"Expected a JSON object, got {type(data).__name__}"

    required_keys = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required_keys:
        if key not in data:
            return f"Missing required key: '{key}'"

    for key, prop_schema in properties.items():
        if key not in data:
            continue
        value = data[key]
        err = _validate_value(key, value, prop_schema)
        if err is not None:
            return err

    return None


def _validate_value(key: str, value: Any, prop_schema: dict[str, Any]) -> str | None:
    expected_type = prop_schema.get("type")

    if expected_type == "integer":
        # bool is a subclass of int but is not an integer value here.
        if isinstance(value, bool) or not isinstance(value, int):
            return f"Key '{key}' expected integer, got {type(value).__name__}"
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"Key '{key}' expected number, got {type(value).__name__}"
        if not math.isfinite(float(value)):
            return f"Key '{key}' must be a finite number, got {value!r}"
    elif expected_type == "string":
        if not isinstance(value, str):
            return f"Key '{key}' expected string, got {type(value).__name__}"
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return f"Key '{key}' expected boolean, got {type(value).__name__}"
    elif expected_type == "array":
        if not isinstance(value, list):
            return f"Key '{key}' expected array, got {type(value).__name__}"
    elif expected_type == "object":
        if not isinstance(value, dict):
            return f"Key '{key}' expected object, got {type(value).__name__}"

    # Range + enum constraints (ignored entirely by the old validators).
    if expected_type in ("number", "integer") and isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = prop_schema.get("minimum")
        maximum = prop_schema.get("maximum")
        if minimum is not None and value < minimum:
            return f"Key '{key}' must be >= {minimum}, got {value!r}"
        if maximum is not None and value > maximum:
            return f"Key '{key}' must be <= {maximum}, got {value!r}"

    enum = prop_schema.get("enum")
    if enum is not None and value not in enum:
        return f"Key '{key}' must be one of {enum}, got {value!r}"

    return None
