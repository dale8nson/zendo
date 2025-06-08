# syntax=docker/dockerfile:1.4

# Stage 1: Build Next.js frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN for i in 1 2 3; do npm ci && break || sleep 5; done
COPY frontend/ .
RUN npm run build . -o out

# Stage 2: Build Rust/Python wheel
FROM python:3.12-slim AS wheel-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl build-essential pkg-config libssl-dev patchelf unzip

# Download minimal libtorch (cannot easily trim more unless you build from source)
RUN curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -o libtorch.zip \
  && unzip libtorch.zip -d /opt \
  && rm libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

RUN pip install maturin

WORKDIR /backend/rust
COPY backend/rust/ .

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN maturin build --release --features python -o /tmp

# Stage 3: Final image
FROM python:3.12-slim AS final

# FastAPI and runtime deps
RUN apt-get update && \
  apt-get install -y --no-install-recommends git curl && \
  pip install --no-cache-dir fastapi uvicorn

# Create Python venv (best practice for Python images)
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv /opt/venv

# Install Python dependencies
WORKDIR /app
COPY backend/python/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source (if this includes junk, filter more tightly)
COPY backend/python/ .

# Copy Rust wheel and install, then remove wheel to save space
COPY --from=wheel-builder /tmp/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy libtorch to final image (required for Rust extension)
COPY --from=wheel-builder /opt/libtorch /opt/libtorch
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Copy static frontend
COPY --from=frontend-builder /frontend/out /app/static

# Copy assets if needed
COPY assets /assets

WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
