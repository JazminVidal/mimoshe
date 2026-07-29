FROM ghcr.io/astral-sh/uv:0.8.3 AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential \
        clang \
        cmake \
        ffmpeg \
        libasound2-dev \
        libopus-dev \
        libssl-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
