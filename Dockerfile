# AskPG app image (CPU). WRITTEN OFFLINE, NOT YET BUILT: the first `docker compose build` is its first test.
# Needs the internet at BUILD time (base image, pip, model downloads); the running container needs none for
# retrieval, only for the hosted LLM.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

COPY requirements.txt .
# Same torch version as the local machine (2.10.0), CPU build. UNVERIFIED that the +cpu wheel of exactly this
# version exists on the index: if the build fails here, read the pip error before changing anything.
RUN pip install --upgrade pip \
&& pip install torch==2.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
&& pip install -r requirements.txt

# Bake the two models into the image so a container starts without downloading anything.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY src ./src
COPY scripts ./scripts
COPY web ./web
COPY data/chunks/chunks.jsonl ./data/chunks/chunks.jsonl

EXPOSE 8000
# ONE worker on purpose (the models and the LLM throttle keep in-process state; see api.py).
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
