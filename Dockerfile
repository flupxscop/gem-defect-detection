# syntax=docker/dockerfile:1

FROM node:20-alpine AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
COPY results/comparison.json ../results/
# The image serves the API too, so the app calls it instead of running models in the browser.
ENV VITE_INFERENCE=server
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Models come from the Hugging Face model repo at a pinned revision (see scripts/publish_models.py).
ARG MODEL_REPO=ChantaroNtw/gemscan-models
ARG MODEL_REVISION=d2fbb1f5c061f0bd458de3fc1d0e9dac9aa456af
ARG MODEL_URL=https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}
RUN mkdir models
ADD --chmod=644 ${MODEL_URL}/manifest.json ${MODEL_URL}/yolo11s.onnx ${MODEL_URL}/rtdetr-l.onnx models/

COPY app/ app/
COPY results/comparison.csv results/
COPY --from=web /build/web/dist web/dist

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
