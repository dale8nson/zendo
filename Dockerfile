# syntax=docker/dockerfile:1

############################################################################
#
# Stage 1: Build the Rust extension (Wheel) with maturin
#
############################################################################
FROM python:3.12-slim AS rust-builder

RUN apt-get update && apt-get install -y \
      curl \
      build-essential \
      pkg-config \
      libssl-dev \
      git \
  && curl https://sh.rustup.rs -sSf | sh -s -- -y \
  && . "$HOME/.cargo/env"

ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install maturin

# Copy your Rust code (assuming "backend/rust" houses your Rust project)
COPY backend/rust /backend/rust

WORKDIR /backend/rust
RUN maturin build --release -o dist
# maturin will produce a .whl inside dist/; we’ll install that in the next stage

############################################################################
#
# Stage 2: Install Python dependencies + the Rust extension wheel
#
############################################################################
FROM python:3.12-slim AS backend-builder

# Copy over the Rust wheel from the previous stage
COPY --from=rust-builder /backend/rust/dist/*.whl /tmp/rust_ext.whl

# Install system dependencies for FastAPI
RUN apt-get update && apt-get install -y \
      git \
      curl \
  && pip install --no-cache-dir fastapi uvicorn

# Install the Rust wheel
RUN pip install /tmp/rust_ext.whl

# Copy Python requirements & code
COPY backend/python/requirements.txt /backend/python/requirements.txt
RUN pip install --no-cache-dir -r /backend/python/requirements.txt

COPY backend/python /backend/python

WORKDIR /backend/python
# We do NOT CMD here, that will be done in the final stage.

############################################################################
#
# Stage 3: Build the Next.js frontend
#
############################################################################
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json   frontend/package-lock.json ./
RUN npm install --production=false

# Copy the rest of the frontend code
COPY frontend ./

# Build Next.js in production mode
RUN npm run build

############################################################################
#
# Stage 4: “runner” → set up a small reverse proxy and run both backend & frontend
#
############################################################################
FROM node:20-alpine AS runner

# 1) Install Python runtime + system deps for backend
RUN apk add --no-cache python3 py3-pip curl \
    && pip3 install --no-cache-dir fastapi uvicorn

# 2) Create /app and copy in the built backend from backend-builder
WORKDIR /app

# Copy the Python/Rust environment from backend-builder
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=backend-builder /backend/python       /app/backend/python

# 3) Copy the built Next.js frontend from frontend-builder
COPY --from=frontend-builder /app/frontend/.next   /app/frontend/.next
COPY --from=frontend-builder /app/frontend/public  /app/frontend/public
COPY --from=frontend-builder /app/frontend/package.json     /app/frontend/package.json
COPY --from=frontend-builder /app/frontend/next.config.js   /app/frontend/next.config.js
COPY --from=frontend-builder /app/frontend/node_modules     /app/frontend/node_modules

# 4) Install a tiny HTTP‐reverse‐proxy in Node to dispatch /api vs frontend
RUN npm install --global http-proxy-middleware@^2.0.6 express@^4.18.2

# 5) Copy over a little “proxy‐server.js” that we will create below
COPY proxy-server.js /app/proxy-server.js

# 6) Expose port 8080 (Fly.io will route HTTP traffic here)
ENV PORT=8080
EXPOSE 8080

# 7) Create a shell script that starts both uvicorn and the Node proxy
COPY start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

CMD ["/app/start-all.sh"]
