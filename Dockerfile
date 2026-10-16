# AskPG app image. Default: CUDA PyTorch 2.5.1+cu124 for NVIDIA GPU inference; build with --build-arg TORCH_VARIANT=cpu for CPU PyTorch.
# Needs the NVIDIA container runtime on the host and a `deploy.resources.reservations.devices`
# block (or `gpus: all`) on the compose services that run models (app, indexer).
# Needs internet at BUILD time for packages and model downloads.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

COPY requirements.txt .

ARG TORCH_VARIANT=cu124
# torch first, once, as a pinned local version (+cu124) with the PyTorch index as an EXTRA index
# (PyPI stays available for torch's dependencies). requirements.txt excludes torch, so it sees
# torch already satisfied and does not pull the default 2.10.0 build over it.
RUN pip install --upgrade pip \
    && pip install torch==2.5.1+${TORCH_VARIANT} --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install -r requirements.txt

# Bake the two models into the image so a container starts without downloading them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY src ./src
COPY scripts ./scripts
COPY web ./web
COPY data/chunks/chunks.jsonl ./data/chunks/chunks.jsonl

EXPOSE 8000
# ONE worker on purpose (the models and the LLM throttle keep in-process state; see api.py).
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]