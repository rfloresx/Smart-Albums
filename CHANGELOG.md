# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-19

### Added

- **Best-of-Year pipeline** — Multi-stage curation: retrieve, score, deduplicate, cluster, and publish a "Best of [Year]" album.
- **Rotation pipeline** — Nightly photo-frame album update with freshness cooldown and temporal balancing.
- **Web GUI** — NiceGUI-based interface for configuring pipelines, managing jobs, prompts, and scheduling.
- **Multiple AI backends** — Ollama, llama.cpp, OpenAI-compatible APIs for vision scoring; HuggingFace or Ollama for embeddings.
- **Perceptual hashing** — 64-bit DCT-based pHash for exact and near-duplicate detection.
- **FAISS similarity search** — Embedding-space near-duplicate grouping via cosine similarity.
- **Scene clustering** — Cosine similarity grouping with MMR-based diverse selection.
- **Balanced selection** — Two-phase algorithm: monthly quota fill then quality top-up.
- **JSONL caching** — Persistent cache for scores and embeddings with automatic compaction on load.
- **Docker images** — CLI one-shot container and WebGUI long-running service.
- **Protocol-based DI** — ProtocolsRegistry enables runtime provider selection via config.
- **Fork strategies** — Parallel fan-out (`Fork`) and cascading remainder (`ForkBySelection`).
- **Config override system** — Alias-based pipeline_settings for per-stage config tuning.

[0.1.0]: https://github.com/rfloresx/Smart-Albums/releases/tag/v0.1.0
