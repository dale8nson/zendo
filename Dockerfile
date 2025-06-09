# syntax=docker/dockerfile:1.4
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN for i in 1 2 3; do npm ci && break || sleep 5; done
COPY frontend/ .
RUN npm run build . -o out

FROM python:3.12-slim AS backend
RUN apt-get update && \
  apt-get install -y --no-install-recommends git curl && \
  rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv /opt/venv

WORKDIR /app

COPY backend/python/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend code + model
COPY backend/python/ .

COPY backend/models/openclip/ .
WORKDIR /app/backend/models/openclip
RUN python script_clip.py

WORKDIR /app

# Debug space used by torch & model
RUN pip show torch torchvision timm open_clip_torch || true
RUN du -h -d 2 /app | sort -hr | head -30

COPY --from=frontend-builder /frontend/out /app/static
COPY assets /assets

RUN du -h -d 3 /app | sort -hr | head -40
RUN find /app -type f -size +100M
RUN du -h -d 2 / | sort -hr | head -40

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
