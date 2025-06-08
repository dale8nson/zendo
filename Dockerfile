# syntax=docker/dockerfile:1.4

# ───── Stage 1: Frontend (Next.js) ─────
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN for i in 1 2 3; do npm ci && break || sleep 5; done
COPY frontend/ .
RUN npm run build . -o out

# ───── Stage 2: Build Rust/Python wheel ─────

FROM python:3.12-slim AS wheel-builder

# Install system deps
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl build-essential pkg-config libssl-dev patchelf unzip

# Download libtorch (CPU, CXX11-ABI)
RUN curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -o libtorch.zip \
  && unzip libtorch.zip -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Install maturin
RUN pip install maturin

# Copy Rust source and manifest
WORKDIR /backend/rust
COPY backend/rust/ .

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Build the wheel
RUN maturin build --release --features python -o /tmp

# ───── Stage 3: Python/Final ─────
FROM python:3.12-slim AS final

# System deps for FastAPI, Uvicorn, etc
RUN apt-get update && \
  apt-get install -y --no-install-recommends git curl && \
  pip install --no-cache-dir fastapi uvicorn

# Node.js (for serving Next.js static if needed)
RUN apt-get update && apt-get install -y curl && \
  curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
  apt-get install -y nodejs

# Python venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv /opt/venv

# Copy and install Python dependencies
WORKDIR /app
COPY backend/python/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy FastAPI app source
COPY backend/python/ .

# Copy the Rust-built wheel from previous stage and install
COPY --from=wheel-builder /tmp/*.whl /tmp/
RUN pip install /tmp/*.whl

# Copy static Next.js build
COPY --from=frontend-builder /frontend/out /app/static
COPY --from=frontend-builder /frontend/public /app/frontend/public
COPY --from=frontend-builder /frontend/package.json /app/frontend/package.json

# Assets
COPY assets /assets

# Entrypoint
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
