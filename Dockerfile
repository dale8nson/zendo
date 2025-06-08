# syntax=docker/dockerfile:1.4

# Stage 1: Build Next.js frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN for i in 1 2 3; do npm ci && break || sleep 5; done
COPY frontend/ .
RUN npm run build . -o out
RUN echo '==== [frontend-builder] Disk usage after build ====' && du -sh /frontend/* || true

# Stage 2: Build Rust/Python wheel
FROM python:3.12-slim AS wheel-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl build-essential pkg-config libssl-dev patchelf unzip
RUN echo '==== [wheel-builder] After system deps ====' && du -sh /* /root/* /opt/* 2>/dev/null || true

RUN curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -o libtorch.zip \
  && unzip libtorch.zip -d /opt \
  && rm libtorch.zip
RUN echo '==== [wheel-builder] After libtorch unzip ====' && du -sh /* /opt/libtorch/* 2>/dev/null || true

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

RUN pip install maturin
RUN echo '==== [wheel-builder] After maturin install ====' && du -sh /* /root/.cache/pip 2>/dev/null || true

WORKDIR /backend/rust
COPY backend/rust/ .
RUN echo '==== [wheel-builder] After Rust source copy ====' && du -sh /backend/rust/* /root/* /opt/* 2>/dev/null || true

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN echo '==== [wheel-builder] After rustup ====' && du -sh /root/.cargo/* 2>/dev/null || true

RUN maturin build --release --features python -o /tmp
RUN echo '==== [wheel-builder] After maturin build ====' && du -sh /tmp/* /root/.cargo/* /opt/libtorch/* 2>/dev/null || true

# Stage 3: Final image
FROM python:3.12-slim AS final

RUN apt-get update && \
  apt-get install -y --no-install-recommends git curl && \
  pip install --no-cache-dir fastapi uvicorn
RUN echo '==== [final] After system deps ====' && du -sh /* /root/* /opt/* 2>/dev/null || true

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv /opt/venv

WORKDIR /app
COPY backend/python/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN echo '==== [final] After pip install requirements ====' && du -sh /opt/venv/* 2>/dev/null || true

COPY backend/python/ .
RUN echo '==== [final] After copying FastAPI app ====' && du -sh /app/* 2>/dev/null || true

COPY --from=wheel-builder /tmp/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
RUN echo '==== [final] After pip install Rust wheel ====' && du -sh /opt/venv/* 2>/dev/null || true

COPY --from=wheel-builder /opt/libtorch /opt/libtorch
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

COPY --from=frontend-builder /frontend/out /app/static
RUN echo '==== [final] After copying frontend ====' && du -sh /app/static 2>/dev/null || true

COPY assets /assets

WORKDIR /app
RUN echo '==== [final] End of build, major dirs ====' && du -sh /app /opt /root /tmp /usr 2>/dev/null || true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
