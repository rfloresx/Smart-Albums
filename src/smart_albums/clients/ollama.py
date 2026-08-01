"""OllamaClient — async wrapper around the official ollama Python library.

Uses ollama.AsyncClient for all interactions with the Ollama server.
Transient errors (ConnectionError) are retried via tenacity with exponential backoff.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ollama import AsyncClient, ResponseError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import ILLMClient, IEmbeddingClient, IHealthCheck

logger = logging.getLogger(__name__)


_retry_policy = retry(
    retry=retry_if_exception_type(ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _validate_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate a dict against a simplified JSON schema.

    Returns None if valid, or an error message string if validation fails.
    Supports checking required keys and basic type constraints from the
    schema's "properties" and "required" fields.
    """
    required_keys = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required_keys:
        if key not in data:
            return f"Missing required key: '{key}'"

    for key, prop_schema in properties.items():
        if key not in data:
            continue
        expected_type = prop_schema.get("type")
        value = data[key]
        if expected_type == "number" and not isinstance(value, (int, float)):
            return f"Key '{key}' expected number, got {type(value).__name__}"
        if expected_type == "string" and not isinstance(value, str):
            return f"Key '{key}' expected string, got {type(value).__name__}"
        if expected_type == "boolean" and not isinstance(value, bool):
            return f"Key '{key}' expected boolean, got {type(value).__name__}"
        if expected_type == "array" and not isinstance(value, list):
            return f"Key '{key}' expected array, got {type(value).__name__}"

    return None

@ProtocolsRegistry.register("ollama", ILLMClient)
@ProtocolsRegistry.register("ollama", IEmbeddingClient)
@ProtocolsRegistry.register("ollama", IHealthCheck)
class OllamaClient:
    """Async client for the Ollama API using the official ollama library.

    Uses ollama.AsyncClient for transport. The library handles image encoding,
    JSON format negotiation, and API endpoint routing internally.

    Usage::

        async with OllamaClient(
            base_url="http://localhost:11434",
            vision_model="llava:latest",
            embed_model="mxbai-embed-large",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)
            embedding = await client.embed(img)

    Can also be used without a context manager — the ollama.AsyncClient is
    created lazily on first use.
    """

    def __init__(
        self,
        base_url: str,
        vision_model: str,
        embed_model: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._vision_model = vision_model
        self._embed_model = embed_model
        self._client: AsyncClient | None = None

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OllamaClient":
        self._client = AsyncClient(host=self._base_url)
        return self

    async def __aexit__(self, *args: object) -> None:
        # The ollama AsyncClient doesn't require explicit cleanup,
        # but we clear the reference for consistency.
        self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> AsyncClient:
        """Return the active client, creating one lazily if needed."""
        if self._client is None:
            self._client = AsyncClient(host=self._base_url)
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

        The image is sent as raw bytes to the ollama chat endpoint via the
        official library. The library handles base64 encoding internally.
        JSON format is requested so the model returns structured output.
        The response is validated against ``return_schema``.

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
        try:
            response = await self._get_client().chat(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_bytes],
                    }
                ],
                format="json",
                think=False,
            )
        except ResponseError as exc:
            # Non-transient API error (e.g. model not found)
            return {"error": f"Ollama API error: {exc}"}
        except (ConnectionError, OSError, TimeoutError) as exc:
            # Transient connection errors — wrap for tenacity to retry
            raise ConnectionError(
                f"Cannot reach Ollama server at {self._base_url}: {exc}"
            ) from exc

        # Extract the content from the response
        raw_content = response.message.content or ""

        if not raw_content.strip():
            return {"error": "Empty content in model response", "raw": ""}

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            return {"error": f"Model response is not valid JSON: {exc}", "raw": raw_content}

        # Validate against the provided schema
        validation_error = _validate_against_schema(parsed, return_schema)
        if validation_error is not None:
            return {"error": f"Schema validation failed: {validation_error}", "raw": parsed}

        result_dict: dict[str, Any] = parsed
        return result_dict

    @_retry_policy
    async def embed(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding vector for an image.

        Sends raw image bytes to the Ollama embed endpoint via the official
        library. The library handles encoding internally.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            The embedding vector as a list of floats.

        Raises:
            ConnectionError: If the Ollama server is unreachable (retried).
            ResponseError: If the Ollama API returns a non-transient error
                (e.g. model not found). Not retried.
        """
        try:
            response = await self._get_client().embed(
                model=self._embed_model,
                input=image_bytes,  # type: ignore[arg-type]
            )
            return [float(x) for x in response.embeddings[0]]
        except ResponseError:
            # Permanent API error (e.g. model not found) — do not wrap as
            # ConnectionError, let it propagate immediately without retry.
            raise
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama server at {self._base_url}: {exc}"
            ) from exc

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embedding vectors for a list of images sequentially.

        The Ollama API processes one image at a time, so this calls embed()
        for each image individually. Images that fail are returned as None.

        Args:
            image_bytes_list: Raw image bytes for each image to embed.

        Returns:
            A list of the same length as input. Each entry is either a
            list[float] embedding vector, or None on failure.
        """
        results: list[list[float] | None] = []
        for image_bytes in image_bytes_list:
            try:
                embedding = await self.embed(image_bytes)
                results.append(embedding)
            except Exception:
                results.append(None)
        return results

    async def close(self) -> None:
        """Release client resources."""
        self._client = None

    @property
    def health_check_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Ollama server via /api/tags.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
