# Lean image for Render FREE tier (512 MB RAM, no card required).
# No sentence-transformers/torch — keeps image small, build fast, RAM tiny.
# Embeddings use the built-in hashing backend; LLM via free Gemini key if set.
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

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=frontend-build /web/dist ./static

RUN mkdir -p /app/backend_data

ENV DATABASE_URL=sqlite:////app/backend_data/career_intel.db \
    LOG_FILE=/app/backend_data/app.log \
    EMBEDDING_BACKEND=hashing

# Render injects $PORT and health-checks /api/health
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
