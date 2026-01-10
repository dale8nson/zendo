#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${BACKEND_PID:-}" ]] && kill $BACKEND_PID || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill $FRONTEND_PID || true
}
trap cleanup INT TERM EXIT

echo "Starting Rust backend..."
(
  cd backend-rs
  RUST_LOG=${RUST_LOG:-info} cargo run
) &
BACKEND_PID=$!

sleep 1
echo "Starting frontend..."
(
  cd frontend
  npm run dev
) &
FRONTEND_PID=$!

echo "Backend PID=$BACKEND_PID, Frontend PID=$FRONTEND_PID"
wait

