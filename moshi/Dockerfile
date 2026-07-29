FROM ghcr.io/astral-sh/uv:debian

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

ENTRYPOINT ["uv", "run", "-p", "3.10", "-m", "moshi.server"]
CMD ["--host", "0.0.0.0", "--port", "8998", "--hf-repo", "kyutai/moshiko-pytorch-bf16"]
