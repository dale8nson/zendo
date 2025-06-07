# syntax=docker/dockerfile:1.4
#
# ──────────────── Stage A: frontend‐builder (Next.js) ────────────────
#
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend

# 1) Copy only package.json & lockfile → install deps (cached until deps change)
COPY frontend/package.json      frontend/package-lock.json    ./
RUN for i in 1 2 3; do npm ci && break || sleep 5; done

# 2) Copy minimal files needed to build
COPY frontend/next.config.ts    ./next.config.ts
COPY frontend/tsconfig.json     ./tsconfig.json
COPY frontend/public/           ./public/
COPY frontend/app/              ./app/
COPY frontend/components/       ./components/
COPY frontend/lib/             ./lib/

# (If you have other folders like “components/” or “styles/”, COPY them here too,
#  before running “npm run build” so that build cache is preserved.)
# 3) Build to a static “out/” directory
RUN   npm run build . -o out


# ──────────────── Stage B: rust `maturin` build ────────────────



# FROM --platform=linux/amd64 python:3.12-slim AS rust-builder

# ENV LIBTORCH=/opt/libtorch
# ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# WORKDIR /backend/rust

# # 1) Install OS‐level deps + rustup + maturin
# RUN --mount=type=cache,id=cargo-registry,target=/root/.cargo/registry \
#   --mount=type=cache,id=cargo-git,target=/root/.cargo/git \
#   apt-get update \
#   && apt-get install -y --no-install-recommends \
#   curl build-essential pkg-config libssl-dev unzip

# RUN --mount=type=cache,id=cargo-registry,target=/root/.cargo/registry \
#   --mount=type=cache,id=cargo-git,target=/root/.cargo/git \
#   curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip -o libtorch.zip \
#   && unzip libtorch.zip -d /opt \
#   && rm libtorch.zip
# ENV LIBTORCH=/opt/libtorch
# ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# COPY backend/rust/Cargo.toml     ./
# COPY backend/rust/Cargo.lock     ./
# COPY backend/rust/src/           ./
# COPY backend/rust/src/lib.rs     ./src/lib.rs
# COPY backend/rust/src/infer.rs   ./src/infer.rs
# COPY backend/rust/src/main.rs    ./src/main.rs
# COPY backend/rust/src/python/    ./src/python/
# COPY backend/rust/src/python/mod.rs    ./src/python/mod.rs
# COPY backend/rust/src/python/python.rs    ./src/python/python.rs

# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# RUN . "$HOME/.cargo/env"
# RUN pip install maturin patchelf
# RUN maturin build --release
# && pip install target/wheels/*.whl
# RUN export PATH="$PATH:/$HOME/.cargo/bin"

# RUN --mount=type=cache,id=cargo-registry,target=/root/.cargo/registry \
#   --mount=type=cache,id=cargo-git,target=/root/.cargo/git \
#   && . "$HOME/.cargo/env" \
#   && pip install maturin \
#   maturin build --release -o dist \
#   && pip install dist/*.whl

# 3) Copy the rest of the Rust sources → rebuild (only if source changed)


# RUN --mount=type=cache,id=cargo-registry,target=/root/.cargo/registry \
#   --mount=type=cache,id=cargo-git,target=/root/.cargo/git \
#   . "$HOME/.cargo/env" && \
#   maturin build --release --features python
# RUN pip install target/wheels/*.whl

# ──────────────── Stage C: python dependencies ────────────────
#
FROM python:3.12-slim AS python-builder
WORKDIR /backend/python

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 1) Copy only requirements.txt → install pip packages (cached until this file changes)
COPY backend/python/requirements.txt   ./requirements.txt
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
  python -m venv /opt/venv && \
  /opt/venv/bin/pip install --upgrade pip && \
  /opt/venv/bin/pip install -r requirements.txt

# 2) Copy the rest of your FastAPI app
COPY backend/python/                   ./

# ──────────────── Stage D: final runtime container ────────────────
#
FROM python:3.12-slim AS final

# COPY --from=frontend-builder /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=frontend-builder /frontend/out /app/static
# 1) Install minimal OS deps + Uvicorn/FastAPI
RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl \
  && pip install --no-cache-dir fastapi uvicorn

RUN apt-get update && apt-get install -y curl \
  && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
  && apt-get install -y nodejs

# 2) Copy the pre-installed Python packages from python-builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3) Copy the maturin‐built wheel (Rust extension) from rust-builder
# COPY --from=rust-builder /usr/local/lib/python3.12/site-packages \
#   /usr/local/lib/python3.12/site-packages

# 4) Copy your Python/FastAPI code
COPY backend/python/                   /app

# 5) Copy the static Next.js output from frontend-builder
COPY --from=frontend-builder /frontend/out /app/static
COPY --from=frontend-builder /frontend/public /app/frontend/public
COPY --from=frontend-builder /frontend/package.json /app/frontend/package.json

# 6) Copy any other assets (e.g. /assets) if needed
COPY assets   /assets

WORKDIR /

# COPY start-all.sh /app/start-all.sh
# RUN chmod +x /app/start-all.sh

# Copy wheel from GitHub Actions artifact (see Note below)
COPY backend/python/app/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl

# 7) Mount static in your app.main:
#    In backend/python/app/main.py you should have something like:
#
#      from fastapi.staticfiles import StaticFiles
#      app = FastAPI()
#      app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
#
#    and you’ve already included your `/api/…` routers under /api/metadata, /api/images, etc.

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
#
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
