"""Concrete client implementations for the smart_albums pipeline."""

from __future__ import annotations

from smart_albums.clients.embedding import OllamaEmbeddingClient
from smart_albums.clients.huggingface_embedding import HuggingFaceEmbeddingClient
from smart_albums.clients.image_embedding import ImageEmbeddingClient
from smart_albums.clients.immich import ImmichClient
from smart_albums.clients.llamacpp import LlamaCppClient
from smart_albums.clients.ollama import OllamaClient
from smart_albums.clients.openai_llm import OpenAIClient

__all__ = [
    "HuggingFaceEmbeddingClient",
    "ImageEmbeddingClient",
    "ImmichClient",
    "LlamaCppClient",
    "OllamaClient",
    "OllamaEmbeddingClient",
    "OpenAIClient",
]
