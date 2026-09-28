# syntax=docker/dockerfile:1
#
# One-shot container image for the immich-smart-albums nightly album rotation
# job (Requirement 14). Each invocation runs exactly ONE rotation and exits;
# this is not a long-running service. An external scheduler (Unraid User
# Scripts cron or host cron) starts the container on the desired cadence.
#
# All persistent state -- the YAML Config_File, the SQLite History_Store, the
# Schedule_State, and the JSONL Analysis_Cache -- lives under the single
# mounted /config appdata directory (Req 14.4), so the image itself is
# stateless and disposable.
#
# Build:
#   docker build -t immich-smart-albums:latest .
#
# Run one rotation (reads /config/rotation.yaml):
#   docker run --rm \
#     -v /mnt/user/appdata/immich-smart-albums:/config \
#     -e IMMICH_URL=http://immich:2283/api \
#     -e IMMICH_API_KEY=your-api-key \
#     -e OLLAMA_URL=http://ollama:11434 \
#     -e OLLAMA_MODEL=llava:latest \
#     immich-smart-albums:latest

# ---------------------------------------------------------------------------
# Stage 1: build a wheel from the src-layout / hatchling project.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tooling for producing the wheel (hatchling build backend).
RUN python -m pip install --no-cache-dir --upgrade pip build

# Copy only what the build backend needs to resolve and package the project.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Produce a wheel under /build/dist.
RUN python -m build --wheel --outdir /build/dist

# ---------------------------------------------------------------------------
# Stage 2: minimal runtime image with the package + dependencies installed.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Keep Python output unbuffered so each invocation's logs reach the container
# log immediately (Req 14.7), and avoid writing .pyc files in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APPDATA_DIR=/config \
    ROTATION_CONFIG=/config/rotation.yaml

# Install the built wheel together with its runtime dependencies.
# git is needed only to resolve the protocols-system git dependency; it is
# purged in the same layer to keep the image small.
COPY --from=builder /build/dist/*.whl /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir "$(ls /tmp/*.whl)[geo]" \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/* /tmp/*.whl

# Single appdata mount point holding the Config_File, History_Store,
# Schedule_State, and Analysis_Cache (Req 14.4). Relative paths in the config
# are resolved against this directory by the CLI.
RUN mkdir -p /config

# Run as a non-root user. The host bind-mount for /config should be owned by
# (or writable to) UID/GID 1000 so the job can persist state across runs.
RUN useradd --uid 1000 --user-group --home-dir /config --no-create-home rotator \
    && chown -R rotator:rotator /config
USER rotator

VOLUME ["/config"]

# Default command runs ONE rotation and exits (Req 14.1). The CLI maps failures
# to a non-zero exit code (Req 14.2/14.3), so a failed run surfaces as an
# exited/failed container. Connection settings come from the env vars above;
# all behavior comes from the mounted config (Req 14.6).
CMD ["smart-albums", "rotation", "--config", "/config/rotation.yaml"]
