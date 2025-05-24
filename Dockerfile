# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# 🛠 Stage 1: Build the Rust Python extension using maturin
FROM base AS builder

# Install networking and system deps (combine for caching)
RUN apt-get update && apt-get install -y \
    dnsutils \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
 && curl https://sh.rustup.rs -sSf | sh -s -- -y \
 && . "$HOME/.cargo/env"

ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin and build wheel
RUN pip install maturin

COPY backend/rust /backend/rust
WORKDIR /backend/rust

RUN maturin build --release -o dist
RUN pip install dist/*.whl

# 🐍 Stage 2: Runtime container
FROM base

# Install Git and Python runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
 && pip install --no-cache-dir fastapi uvicorn

# Install Python dependencies
COPY backend/python/requirements.txt /backend/python/requirements.txt
RUN pip install --no-cache-dir -r /backend/python/requirements.txt

# Copy app code and assets
COPY backend/python /backend/python
COPY assets /assets

# Copy Rust-built site-packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

WORKDIR /backend/python

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
