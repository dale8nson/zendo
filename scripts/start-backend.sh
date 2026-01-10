#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load project root .env if present
set +u
if [ -f "$ROOT_DIR/.env" ]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
fi
set -u

# Default SDXL path if not provided
export ZENDO_SDXL_DIR="${ZENDO_SDXL_DIR:-$ROOT_DIR/backend/models/sdxl-base-1.0/ONNX}"

# Try to auto-detect ONNX Runtime dylib if not provided
if [ -z "${ORT_DYLIB_PATH:-}" ] && [ -z "${ZENDO_ORT_DYLIB_PATH:-}" ]; then
  # Prefer Python wheel if installed
  PY_BIN="${PYTHON:-python3}"
  CAND="$($PY_BIN - <<'PY' 2>/dev/null || true
try:
    import onnxruntime, pathlib
    base = pathlib.Path(onnxruntime.__file__).parent
    for pat in ("**/libonnxruntime.*", "libonnxruntime.*"):
        matches = list(base.glob(pat))
        if matches:
            print(matches[0])
            break
except Exception:
    pass
PY
)"
  if [ -n "$CAND" ] && [ -f "$CAND" ]; then
    export ORT_DYLIB_PATH="$CAND"
  else
    # Fallback to common Homebrew locations
    for p in /opt/homebrew/lib/libonnxruntime.dylib /usr/local/lib/libonnxruntime.dylib; do
      if [ -f "$p" ]; then
        export ORT_DYLIB_PATH="$p"
        break
      fi
    done
  fi
fi

cd "$ROOT_DIR/backend-rs"

## Choose cargo features for ONNX EPs. Default to CoreML on Apple Silicon if not specified.
FEATURES_ARG=""
if [ -n "${ZENDO_ORT_FEATURES:-}" ]; then
  FEATURES_ARG="--features ${ZENDO_ORT_FEATURES}"
else
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    FEATURES_ARG="--features coreml"
  fi
fi

echo "Starting backend with cargo features: ${FEATURES_ARG:-<none>}"
RUST_LOG=${RUST_LOG:-info} cargo run ${FEATURES_ARG}
