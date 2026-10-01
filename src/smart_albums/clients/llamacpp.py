"""LlamaCppClient — async HTTP client for llama.cpp server (OpenAI-compatible API).

Communicates with a llama.cpp server via its OpenAI-compatible chat/completions
endpoint for vision tasks. Supports structured JSON output via the grammar or
response_format parameter. Transient connection errors are retried via tenacity.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import ILLMClient, IHealthCheck

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


@ProtocolsRegistry.register("llamacpp", ILLMClient)
@ProtocolsRegistry.register("llamacpp", IHealthCheck)
class LlamaCppClient:
    """Async client for a llama.cpp server via its OpenAI-compatible API.

    Connects to a llama.cpp server (started with ``--port`` and optionally
    ``--api-key``) using the ``/v1/chat/completions`` endpoint for vision
    tasks and ``/v1/embeddings`` for embedding generation.

    The server must be loaded with a multimodal model (e.g. LLaVA, Qwen-VL)
    for ``analyze_image`` to work. For embeddings, the server should be
    started with ``--embedding`` enabled.

    Usage::

        async with LlamaCppClient(
            base_url="http://localhost:8080",
            vision_model="qwen-vl",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)
            embedding = await client.embed(img)

    Args:
        base_url: The llama.cpp server URL (e.g. "http://localhost:8080").
        vision_model: Model identifier (used for logging; llama.cpp serves
            a single model so this is informational).
        api_key: Optional API key if the server was started with ``--api-key``.
        timeout: HTTP request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        base_url: str,
        vision_model: str = "default",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._vision_model = vision_model
        self._api_key = api_key
        self._timeout = float(timeout)
        self._client: httpx.AsyncClient | None = None

        if vision_model == "default":
            logger.warning(
                "LlamaCppClient created with vision_model='default'. "
                "Set 'vision_model' in config to suppress this warning. "
                "Note: llama.cpp serves whichever model it was started with "
                "regardless of this value."
            )

    @property
    def vision_model(self) -> str:
        return self._vision_model

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LlamaCppClient":
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client, creating one lazily if needed."""
        if self._client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout),
            )
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

        Encodes the image as base64 and sends it to the llama.cpp server's
        OpenAI-compatible chat/completions endpoint with a JSON response
        format constraint. The response is validated against ``return_schema``.

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
        # Normalize to a format the vision endpoint accepts. Unlike the old
        # inline code — which only converted WebP, mislabeled everything
        # else as image/jpeg while sending the original bytes, and on a
        # conversion failure still sent the bad bytes — this converts every
        # non-JPEG/PNG format (HEIC/TIFF/BMP/GIF/WebP), downscales large
        # images, and fails cleanly if conversion isn't possible (CL-08).
        try:
            image_bytes, mime_type = _prepare_image_for_vision(image_bytes)
        except _ImageConversionError as exc:
            return {"error": f"Image conversion failed: {exc}"}

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        logger.debug(
            "analyze_image: sending %d bytes as %s to model %s",
            len(image_bytes), mime_type, self._vision_model,
        )

        payload: dict[str, Any] = {
            "model": self._vision_model,
            "messages": [
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
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "cache_prompt": True,
        }

        try:
            response = await self._get_client().post(
                "/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"llama.cpp server error (HTTP {exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc
            return {"error": f"llama.cpp API error (HTTP {exc.response.status_code}): {exc.response.text}"}
        except httpx.ConnectTimeout as exc:
            # Couldn't even establish the connection — safe and worthwhile
            # to retry.
            raise ConnectionError(
                f"Timeout connecting to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            # A read timeout means the request *was* sent and the server is
            # (still) processing it — retrying would pile a second
            # expensive inference on top of the first while the abandoned
            # one keeps running (CL-09). Return a per-image error instead of
            # raising ConnectionError (which tenacity would retry 3x).
            return {
                "error": (
                    f"llama.cpp request timed out after {self._timeout}s "
                    f"(server may still be processing): {exc}"
                )
            }
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            # Transport-level errors (connection reset mid-stream, protocol
            # violation) previously escaped unhandled and bubbled all the
            # way out of the stage (CL-09). Treat them as transient and
            # retryable.
            raise ConnectionError(
                f"Transport error talking to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            # Any other timeout type — be conservative and don't retry a
            # possibly-still-running inference.
            return {"error": f"llama.cpp request timed out: {exc}"}

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON in server response: {exc}", "raw": response.text}

        # Extract content from OpenAI-compatible response
        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            return {"error": f"Unexpected response structure: {exc}", "raw": data}

        if not raw_content or not raw_content.strip():
            return {"error": "Empty content in model response", "raw": ""}

        # Strip markdown code fences if the model wrapped its JSON output
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

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

        Sends the image as base64 to the llama.cpp server's ``/v1/embeddings``
        endpoint. The server must be running with ``--embedding`` enabled
        **and** loaded with a multimodal embedding model.

        .. warning::
            ``/v1/embeddings`` on a text-only embedding model will embed the
            request payload as text rather than the image's visual content,
            silently producing a vector unrelated to the picture (CL-09).
            Only use this against a server you know is serving a multimodal
            embedding model; otherwise prefer a dedicated image-embedding
            provider (``huggingface``, ``image_embedding``).

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            The embedding vector as a list of floats.

        Raises:
            ConnectionError: If the server is unreachable (retried).
            RuntimeError: If the server returns an error response.
        """
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload: dict[str, Any] = {
            "input": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                }
            ],
            "model": self._vision_model,
        }

        try:
            response = await self._get_client().post(
                "/v1/embeddings",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"llama.cpp server error (HTTP {exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc
            raise RuntimeError(
                f"llama.cpp embedding error (HTTP {exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            # Transport-level errors previously escaped unhandled (CL-09).
            raise ConnectionError(
                f"Transport error talking to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Timeout connecting to llama.cpp server at {self._base_url}: {exc}"
            ) from exc

        data = response.json()
        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected embedding response structure: {exc}"
            ) from exc

        return [float(x) for x in embedding]

    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def health_check_url(self) -> str:
        """The llama.cpp server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the llama.cpp server.

        Tries /health first (native llama.cpp), then falls back to
        /v1/models (OpenAI-compatible endpoint available on all variants).

        Returns:
            True if the server responds successfully, False otherwise.
        """
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                # Try /health first (native llama-server)
                resp = await client.get(f"{self._base_url}/health")
                if resp.status_code == 200:
                    return True
                # Fall back to OpenAI-compatible /v1/models
                resp = await client.get(f"{self._base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False
