# Copyright 2026 Query Farm LLC - https://query.farm
#
# Single image serving BOTH transports of the vgi-audio worker:
#   docker run ... IMG            -> HTTP server on $PORT (default 8000; /health, VGI RPC)
#   docker run -i ... IMG stdio   -> stdio worker DuckDB spawns on-host
# See docker-entrypoint.sh. Audio decodes natively for WAV/FLAC/OGG (libsndfile via
# soundfile); compressed formats (mp3/m4a) need ffmpeg, which is intentionally not
# bundled — those decodes return NULL rather than crashing.
# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Query-farm/vgi-audio

LABEL org.opencontainers.image.title="vgi-audio" \
      org.opencontainers.image.description="librosa audio-feature extraction (duration, tempo, MFCC, key, ...) for DuckDB via VGI (stdio + HTTP)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      farm.query.vgi.transports='["http","stdio"]'

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# libsndfile1 backs soundfile's native WAV/FLAC/OGG decode; curl backs the
# HEALTHCHECK and the CI /health smoke.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install the worker + HTTP-serving extra from the source tree.
COPY pyproject.toml README.md LICENSE ./
COPY vgi_audio ./vgi_audio
RUN pip install '.[serve]'

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
