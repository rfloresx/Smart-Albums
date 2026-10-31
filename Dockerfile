# syntax=docker/dockerfile:1
#
# Single multi-purpose image for smart-albums. It serves BOTH deployment
# shapes that previously needed two separate Dockerfiles (DK-04 /
# SIMPLIFICATION 4.2):
#
#   * Long-running web GUI (default CMD): NiceGUI service on port 8080.
#       docker run --rm -p 127.0.0.1:8080:8080 \
#         -v ./config.yaml:/config/config.yaml:ro \
#         -v smart-albums-data:/data \
#         smart-albums:latest
#
#   * One-shot rotation job (override the command): runs one rotation and
#     exits, for an external cron/scheduler to invoke on a cadence.
#       docker run --rm \
#         -v /path/to/appdata:/config \
#         -e IMMICH_URL=http://immich:2283/api -e IMMICH_API_KEY=... \
#         -e OLLAMA_URL=http://ollama:11434 -e OLLAMA_MODEL=llava:latest \
#         smart-albums:latest smart-albums rotation --config /config/rotation.yaml
#
# Both entrypoints share the same wheel and dependency set, mirroring the
# way PhotoStudio's compose profiles share one image.

# ---------------------------------------------------------------------------
# Stage 1: build a wheel from the src-layout / hatchling project.
# ---------------------------------------------------------------------------
# Base image pinned by digest for reproducible builds (DK-04), using the same
# verified python:3.12-slim index digest as the PhotoStudio image so every
# image in the workspace tracks one base. To bump:
#   docker buildx imagetools inspect python:3.12-slim   # copy the index digest
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS builder

WORKDIR /build

# Pin build tooling versions so a wheel built today and one built next month
# use the same toolchain (DK-04).
RUN python -m pip install --no-cache-dir --upgrade "pip==24.2" "build==1.2.2"

# Copy only what the build backend needs.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Produce a wheel under /build/dist.
RUN python -m build --wheel --outdir /build/dist

# ---------------------------------------------------------------------------
# Stage 2: runtime image with the package + all entrypoints' dependencies.
# ---------------------------------------------------------------------------
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS runtime

WORKDIR /app

# PORT standardized to 8080 across all workspace images (SIMPLIFICATION 4.2).
# The webgui reads PORT from the environment (default 8000 for local runs).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=/config/config.yaml \
    ROTATION_CONFIG=/config/rotation.yaml \
    HF_HOME=/data/hf_cache \
    PORT=8080

# System dependencies for imagehash/Pillow/faiss and the webgui.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (avoids pulling the ~2GB CUDA variant). Pinned so
# the image is reproducible rather than tracking whatever the CPU index's
# latest torch happens to be at build time (DK-04). Needed by the
# huggingface embedding provider.
RUN pip install --no-cache-dir \
    "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu

# Install the built wheel with every extra both entrypoints may use. torch is
# already satisfied above so pip won't re-download it. No git is needed:
# protocols-system is now a published versioned wheel, not a git dependency
# (SIMPLIFICATION 4.2), so the whole package resolves from the index.
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[geo,huggingface,openai,webgui,heif]" \
    && rm -rf /tmp/*.whl

# Copy the maintenance scripts and default prompts only. The installed
# package (from the wheel) is the single source of truth for code; copying
# src/webgui over it would shadow the installed tree (DK-03).
COPY scripts /app/scripts
COPY prompts /app/prompts
ENV SMART_ALBUMS_SCRIPTS_DIR=/app/scripts

# Create data/config directories up front so that when compose mounts named
# volumes over them they already exist owned by appuser (Docker otherwise
# creates missing mount points as root, breaking writes). /config also serves
# the one-shot rotation job's single appdata mount.
RUN mkdir -p /data/cache /data/prompts /data/db /data/hf_cache /data/exports /config

# Run as non-root. Only the writable data/config volumes are chowned to
# appuser — NOT /app — so a compromised request handler can't rewrite the
# code it's running (DK-03). UID/GID 1000 so a host bind-mount for the
# rotation job's /config can be made writable to it.
RUN useradd --uid 1000 --user-group --no-create-home appuser \
    && chown -R appuser:appuser /data /config
USER appuser

EXPOSE 8080

# Healthcheck for the long-running webgui so `restart: unless-stopped` can
# tell a wedged server from a healthy one (DK-05). Hits the login page, which
# is always served (reachable without auth) and returns 200 once up. For the
# one-shot rotation command the healthcheck is irrelevant (the container
# exits), and compose jobs that run it don't enable it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/login', timeout=4).status==200 else 1)" || exit 1

# Default: run the long-running web GUI. Override with
# `smart-albums rotation --config /config/rotation.yaml` for the one-shot job.
CMD ["python", "-m", "webgui"]
