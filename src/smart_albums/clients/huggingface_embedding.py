"""HuggingFace local embedding client for the smart_albums pipeline.

Provides a concrete implementation of the IEmbeddingClient protocol using
a local HuggingFace Transformers model (AutoModel + AutoProcessor) for
computing vector embeddings from raw image bytes.

Uses the same AutoModel approach as the original immich_smart_albums backend:
any CLIP-family model can be loaded by its HuggingFace Hub ID
(e.g. "openai/clip-vit-base-patch32", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K").

Two embedding APIs are provided:

* ``embed(image_bytes)`` — single-image path; satisfies the base
  ``IEmbeddingClient`` protocol and is used as a fallback by the node layer.
* ``embed_batch(image_bytes_list)`` — processes a list of images in batched
  forward passes (``batch_size`` images per pass). On GPU this is significantly
  faster than calling ``embed()`` N times because it amortises data-transfer and
  kernel-launch overhead. The node layer detects this method via ``hasattr`` and
  prefers it when available.

This module requires the optional ``huggingface`` dependency group:
    pip install immich-smart-albums[huggingface]

Heavy imports (torch, transformers, PIL) are deferred until runtime
to keep the module importable without those packages installed.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IEmbeddingClient, IHealthCheck

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Verify that optional HuggingFace dependencies are installed.

    Raises:
        ImportError: With a clear message if transformers or torch
            are not available.
    """
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceEmbeddingClient requires the 'transformers' package. "
            "Install it with: pip install immich-smart-albums[huggingface]"
        ) from exc

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceEmbeddingClient requires the 'torch' package. "
            "Install it with: pip install immich-smart-albums[huggingface]"
        ) from exc


@ProtocolsRegistry.register("huggingface", IEmbeddingClient)
@ProtocolsRegistry.register("huggingface", IHealthCheck)
class HuggingFaceEmbeddingClient:
    """Async embedding client using HuggingFace AutoModel + AutoProcessor.

    Implements the IEmbeddingClient protocol by loading any CLIP-family
    (or compatible vision) model via AutoModel from the HuggingFace Hub
    and encoding images into vector embeddings.

    For CLIP/SigLIP models, ``get_image_features()`` is called directly.
    For generic vision models the forward pass output is pooled. This
    mirrors the original immich_smart_albums HuggingFaceEmbeddingBackend.

    Since model inference is synchronous and CPU/GPU-bound, calls are
    dispatched via ``asyncio.to_thread`` to avoid blocking the event loop.

    Supports async context manager protocol for model lifecycle management.
    The model is loaded in ``__aenter__`` and released in ``__aexit__``.

    Usage — single image::

        async with HuggingFaceEmbeddingClient(
            model_name="openai/clip-vit-base-patch32",
            device="auto",
        ) as client:
            embedding = await client.embed(image_bytes)

    Usage — batch (preferred for throughput)::

        async with HuggingFaceEmbeddingClient(
            model_name="openai/clip-vit-base-patch32",
            device="auto",
            batch_size=16,
        ) as client:
            # Returns list[list[float] | None]; None for images that failed to decode
            embeddings = await client.embed_batch([img1, img2, img3])

    Args:
        model_name: HuggingFace Hub model ID
            (e.g. ``"openai/clip-vit-base-patch32"``).
        device: Compute device: ``"cpu"``, ``"cuda"``, or ``"auto"``
            (auto selects CUDA when available, falls back to CPU).
        batch_size: Maximum number of images per forward pass in
            ``embed_batch``. Larger values increase GPU utilisation but
            also peak VRAM usage. Defaults to 16.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device_arg = device
        self._batch_size = int(batch_size)
        # Resolved at load time
        self._device: str | None = None
        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None
        self._has_get_image_features: bool = False

    @property
    def embed_model(self) -> str:
        """The HuggingFace model ID."""
        return self._model_name

    @property
    def device(self) -> str | None:
        """The resolved torch device string (available after __aenter__)."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Images per forward pass used by embed_batch."""
        return self._batch_size

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HuggingFaceEmbeddingClient:
        """Load the model and processor on the configured device."""
        _check_dependencies()
        await asyncio.to_thread(self._load_model)
        logger.info(
            "Loaded HuggingFace model '%s' on device '%s'",
            self._model_name,
            self._device,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Release model and free GPU memory."""
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load AutoModel + AutoProcessor synchronously.

        Resolves the device, loads both the processor and the model,
        and sets the model to eval mode. Gradient computation is disabled
        per-call via ``torch.inference_mode()`` in ``_run_forward``.

        Raises:
            RuntimeError: If CUDA is requested explicitly but unavailable,
                or if the model cannot be loaded.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        # Resolve device
        if self._device_arg == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._device_arg == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Use device='cpu' or device='auto' instead."
                )
            self._device = "cuda"
        else:
            self._device = "cpu"

        try:
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self._model_name}': {exc}"
            ) from exc

        # Detect CLIP/SigLIP — these expose get_image_features() directly
        self._has_get_image_features = hasattr(self._model, "get_image_features")

    def _run_forward(self, images: list[PILImage.Image]) -> list[list[float]]:
        """Run a single batched forward pass on a list of PIL images.

        All images must already be decoded. The processor handles padding
        and resizing. Returns one embedding vector per input image.

        Uses ``torch.inference_mode()`` to disable gradient computation and
        autograd overhead for the forward pass.

        Args:
            images: Non-empty list of PIL Image objects (RGB).

        Returns:
            List of embedding vectors, one per image, as lists of floats.

        Raises:
            RuntimeError: If the model is not loaded or output is unrecognised.
        """
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )

        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if hasattr(v, "to")
        }

        with torch.inference_mode():
            if self._has_get_image_features:
                pixel_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}
                outputs = self._model.get_image_features(**pixel_inputs)
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "image_embeds"):
                    outputs = outputs.image_embeds
            else:
                outputs = self._model(**inputs)
                if hasattr(outputs, "image_embeds"):
                    outputs = outputs.image_embeds
                elif hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    outputs = outputs.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError(
                        f"Model '{self._model_name}' output has no recognised "
                        "embedding attribute (image_embeds, pooler_output, "
                        "last_hidden_state)."
                    )

        # outputs shape: (batch, embedding_dim)
        return [[float(x) for x in row] for row in outputs.detach().cpu()]

    def _encode_image(self, image_bytes: bytes) -> list[float]:
        """Run synchronous inference for a single image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            The embedding vector as a list of floats.

        Raises:
            RuntimeError: If the model has not been loaded, or if the
                model output has no recognised embedding attribute.
        """
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._run_forward([image])[0]

    def _encode_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Run batched synchronous inference over a list of raw image bytes.

        Images that fail to decode are returned as ``None`` at their
        original index so callers can map results back to asset IDs
        without re-sorting.

        Processes images in chunks of ``self._batch_size`` per forward
        pass so VRAM usage stays bounded regardless of input size.

        Args:
            image_bytes_list: Raw image bytes for each asset.

        Returns:
            List of the same length as ``image_bytes_list``. Each entry is
            either the embedding vector (list of floats) or ``None`` when
            the image could not be decoded.
        """
        from PIL import Image

        # Decode all images, preserving original indices
        decoded: list[tuple[int, Image.Image]] = []
        results: list[list[float] | None] = [None] * len(image_bytes_list)

        for idx, image_bytes in enumerate(image_bytes_list):
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                decoded.append((idx, img))
            except Exception as exc:
                logger.warning(
                    "Failed to decode image at batch index %d: %s", idx, exc
                )

        if not decoded:
            return results

        # Process in chunks of batch_size
        for chunk_start in range(0, len(decoded), self._batch_size):
            chunk = decoded[chunk_start : chunk_start + self._batch_size]
            indices = [idx for idx, _ in chunk]
            images = [img for _, img in chunk]

            try:
                embeddings = self._run_forward(images)
                for idx, embedding in zip(indices, embeddings):
                    results[idx] = embedding
            except Exception as exc:
                logger.warning(
                    "Forward pass failed for batch chunk starting at "
                    "index %d: %s",
                    chunk_start,
                    exc,
                )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Compute an embedding vector for a single image.

        Dispatches synchronous model inference to a thread pool so the
        async event loop is not blocked.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            The embedding vector as a list of floats.

        Raises:
            RuntimeError: If the model has not been loaded via __aenter__.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )
        return await asyncio.to_thread(self._encode_image, image_bytes)

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embedding vectors for a list of images in batched forward passes.

        Preferred over calling ``embed()`` in a loop when processing many
        images, especially on GPU, because the processor pads images into a
        single tensor and the forward pass amortises kernel-launch and
        data-transfer overhead across the whole batch.

        Images that fail to decode are returned as ``None`` at their original
        index so the caller can handle failures without disrupting the rest.

        Args:
            image_bytes_list: Raw image bytes for each image to embed.

        Returns:
            A list of the same length as ``image_bytes_list``. Each entry is
            either a ``list[float]`` embedding vector, or ``None`` if that
            image failed to decode.

        Raises:
            RuntimeError: If the model has not been loaded via __aenter__.

        Example::

            results = await client.embed_batch([img_a, img_b, img_c])
            for i, vec in enumerate(results):
                if vec is None:
                    print(f"image {i} failed")
                else:
                    print(f"image {i}: dim={len(vec)}")
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )
        if not image_bytes_list:
            return []
        return await asyncio.to_thread(self._encode_batch, image_bytes_list)

    async def close(self) -> None:
        """Release the model, processor, and GPU memory.

        After calling close the client cannot be used for embedding
        until ``__aenter__`` is called again.
        """
        self._model = None
        self._processor = None

        if self._device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._device = None
        logger.info("Released HuggingFace model '%s'", self._model_name)

    @property
    def health_check_url(self) -> str:
        """The model name (local provider, no URL)."""
        return f"local://{self._model_name}"

    async def health_check(self) -> bool:
        """Check if the local model is loaded and ready.

        Returns:
            True if the model and processor are loaded, False otherwise.
        """
        return self._model is not None and self._processor is not None
