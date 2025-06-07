# syntax=docker/dockerfile:1.4

FROM node:18-alpine AS frontend-builder
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ──────────────── Python builder ────────────────
FROM python:3.12-slim AS python-builder
WORKDIR /backend/python

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY backend/python/requirements.txt ./
RUN python -m venv /opt/venv && pip install --upgrade pip && pip install -r requirements.txt

COPY backend/python/ ./

# ──────────────── Final runtime ────────────────
FROM python:3.12-slim AS final
WORKDIR /app

# Install OS deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy frontend
COPY --from=frontend-builder /frontend/out ./static

# Copy Python env
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend app
COPY backend/python/ ./app
COPY backend/python/app/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
