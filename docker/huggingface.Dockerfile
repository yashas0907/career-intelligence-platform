# Hugging Face Spaces (FREE Docker hosting, 2 vCPU / 16 GB, auto-deploy from GitHub)
# Build from repo root — same as deploy/Dockerfile but reads the HF port.
FROM node:20-alpine AS frontend-build

WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentence-transformers || echo "ST skipped; hashing/OpenAI backends remain available"

COPY backend/app ./app
COPY --from=frontend-build /web/dist ./static

RUN mkdir -p /app/backend_data

# HF Spaces convention: the app must listen on $PORT (default 7860)
ENV DATABASE_URL=sqlite:////app/backend_data/career_intel.db \
    LOG_FILE=/app/backend_data/app.log

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
