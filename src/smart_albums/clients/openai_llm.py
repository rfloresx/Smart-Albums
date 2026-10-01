"""OpenAIClient — async LLM client using the official openai Python library.

Supports any OpenAI-compatible API (OpenAI, Azure, local proxies) for
vision analysis. Uses the official ``openai.AsyncOpenAI`` client which
handles auth, retries, and streaming internally.

Transient connection errors are retried via tenacity with exponential backoff.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IHealthCheck, ILLMClient

from smart_albums.clients._llm_support import (
    ImageConversionError as _ImageConversionError,
    prepare_image_for_vision as _prepare_image_for_vision,
    validate_against_schema as _validate_against_schema,
)

logger = logging.getLogger(__name__)


_retry_policy = retry(
    retry=retry_if_exception_type(ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model output if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    return cleaned


@ProtocolsRegistry.register("openai", ILLMClient)
@ProtocolsRegistry.register("openai", IHealthCheck)
class OpenAIClient:
    """Async client for OpenAI-compatible APIs using the official openai library.

    Supports vision analysis via GPT-4o, GPT-4-turbo, or any model that
    accepts image inputs. Also works with OpenAI-compatible endpoints
    (e.g. Azure OpenAI, vLLM, LiteLLM) by setting a custom ``base_url``.

    Usage::

        async with OpenAIClient(
            api_key="sk-...",
            vision_model="gpt-4o",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)

    Args:
        api_key: OpenAI API key (or compatible service key).
        vision_model: Model ID for vision tasks (e.g. "gpt-4o", "gpt-4o-mini").
        base_url: Optional custom base URL for OpenAI-compatible APIs.
            Defaults to OpenAI's production endpoint.
        timeout: Request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        api_key: str = "",
        vision_model: str = "gpt-4o",
        base_url: str = "",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._vision_model = vision_model
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = float(timeout)
        self._client: Any = None  # openai.AsyncOpenAI

        if not api_key:
            logger.warning(
                "OpenAIClient created without an api_key. "
                "Set 'api_key' in your LLM provider config."
            )

    @property
    def vision_model(self) -> str:
        return self._vision_model

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenAIClient":
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self._timeout,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = AsyncOpenAI(**kwargs)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the active OpenAI client, creating one lazily if needed."""
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_retry_policy
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        return_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an image to the vision model and validate the response.

        Encodes the image as base64 and sends it via the chat completions
        API with JSON response format. The response is validated against
        ``return_schema``.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
            prompt: Text prompt for the vision model.
            return_schema: JSON-schema-like dict describing expected response structure.

        Returns:
            On success: the parsed JSON dict matching the schema.
            On JSON parse failure: {"error": "<message>", "raw": "<raw_text>"}.
            On schema validation failure: {"error": "<message>", "raw": <parsed_json>}.
            On API error: {"error": "<message>"}.

        Raises:
            ConnectionError: On transient connection errors (retried by tenacity).
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        # Normalize to JPEG/PNG. Previously only WebP was converted and
        # every other non-JPEG/PNG format (HEIC/TIFF/BMP/GIF) was sent as
        # ``data:image/jpeg`` carrying its original, non-JPEG bytes; a
        # conversion failure logged a warning and sent the bad bytes anyway
        # (CL-08). Now everything that isn't already JPEG/PNG is decoded,
        # downscaled, and re-encoded, and an undecodable image returns a
        # clean per-image error instead of being sent as garbage.
        try:
            image_bytes, mime_type = _prepare_image_for_vision(image_bytes)
        except _ImageConversionError as exc:
            return {"error": f"Image conversion failed: {exc}"}

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ],
            }
        ]

        try:
            response = await self._get_client().chat.completions.create(
                model=self._vision_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except APIConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach OpenAI API: {exc}"
            ) from exc
        except APITimeoutError as exc:
            raise ConnectionError(
                f"OpenAI API timeout: {exc}"
            ) from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise ConnectionError(
                    f"OpenAI server error (HTTP {exc.status_code}): {exc.message}"
                ) from exc
            return {"error": f"OpenAI API error (HTTP {exc.status_code}): {exc.message}"}

        raw_content = response.choices[0].message.content or ""

        if not raw_content.strip():
            return {"error": "Empty content in model response", "raw": ""}

        # Strip markdown fences if present
        cleaned = _strip_markdown_fences(raw_content)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {"error": f"Model response is not valid JSON: {exc}", "raw": raw_content}

        validation_error = _validate_against_schema(parsed, return_schema)
        if validation_error is not None:
            return {"error": f"Schema validation failed: {validation_error}", "raw": parsed}

        result_dict: dict[str, Any] = parsed
        return result_dict

    @_retry_policy
    async def embed(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding vector for an image.

        Note: OpenAI's embedding API does not natively support image input.
        This method is provided for protocol compliance but will raise
        RuntimeError. Use a dedicated embedding provider instead.

        Raises:
            RuntimeError: Always — OpenAI embeddings don't support images.
        """
        raise RuntimeError(
            "OpenAI embedding API does not support image input. "
            "Use a dedicated embedding provider (image_embedding, huggingface, ollama)."
        )

    @property
    def health_check_url(self) -> str:
        """The OpenAI API base URL."""
        return self._base_url or "https://api.openai.com"

    async def health_check(self) -> bool:
        """Check connectivity to the OpenAI API by listing models.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        try:
            client = self._get_client()
            await client.models.list()
            return True
        except (APIConnectionError, APITimeoutError, APIStatusError):
            return False
        except Exception:
            return False

    async def close(self) -> None:
        """Release the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
