# syntax=docker/dockerfile:1.7
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HH_RAG_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY constraints-docker.txt ./
COPY src ./src
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      --constraint constraints-docker.txt "torch==2.13.0+cpu" \
    && python -m pip install --constraint constraints-docker.txt ".[app,web]"

# The configuration is tracked; large model/index/data artifacts are mounted at runtime.
# This keeps image builds reproducible from a clean checkout while allowing deployments to
# provision artifacts through an image layer, volume, or artifact store.
COPY results/final_retriever_config.json ./results/final_retriever_config.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "-m", "uvicorn", "hh_goa_rag.web:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
