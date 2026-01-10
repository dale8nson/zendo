# Zendo Rust Backend (Axum + ONNX Runtime)

- Axum server with REST and WebSocket endpoints
- SDXL inference over WebSocket with progress streaming
- ONNX Runtime integration (CPU by default; optional GPU EPs via features)
- Provider scaffolding for CUDA, TensorRT, CoreML, DirectML, OpenVINO

## Build

```
cd backend-rs
# CPU only
cargo run

# With CUDA (ensure ORT with CUDA EP is available and `LD_LIBRARY_PATH` set)
cargo run --features cuda

# With CoreML on macOS (Apple Silicon)
cargo run --features coreml
```

By default the binary dynamically loads ONNX Runtime (`ort`) from your system. Set `ORT_DYLIB_PATH` to point to the shared library if it isn’t on the loader path.

## Run

Env vars:
- `ZENDO_BIND_ADDR` (default `0.0.0.0:8080`)
- `ZENDO_SDXL_DIR` (path to SDXL ONNX models; optional in this scaffold)
- `ZENDO_CONCURRENCY` (overrides workers; default = logical CPUs)
- `ZENDO_STEPS` (default steps per request; default 30)
- `ZENDO_UPSCALER_ONNX` (path to Real-ESRGAN x4 ONNX model; enables /api/upscale). If omitted, the server auto-detects `backend/models/Real-ESRGAN/RealESRGAN_x4plus.onnx` if present.

```
RUST_LOG=info ZENDO_BIND_ADDR=127.0.0.1:8080 cargo run
```

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `GET /info` → build and feature info
- `POST /api/upscale` → stubbed (501)
  - If `ZENDO_UPSCALER_ONNX` is set to a valid ONNX file, runs 4x upscaling and returns a PNG data URL
- `GET /ws/sdxl` → WebSocket for SDXL generation

### WebSocket Protocol (`/ws/sdxl`)
Client → Server:
```
{"type":"generate","request_id":"<optional-uuid>","params":{
  "prompt":"Astronaut riding a horse",
  "negative_prompt":"",
  "seed":123,
  "width":1024,
  "height":1024,
  "steps":30,
  "guidance_scale":7.5
}}
```

Server → Client (examples):
```
{"type":"ready"}
{"type":"progress","request_id":"...","step":5,"total":30}
{"type":"result","request_id":"...","image_base64":"data:image/png;base64,...","seed":123,"duration_ms":5400}
{"type":"error","request_id":"...","message":"..."}
```

## Notes on ONNX Runtime EPs

- This scaffold uses the `ort` crate with dynamic loading. To use GPU EPs you must:
  - Build ONNX Runtime with the desired EPs (CUDA/TensorRT/CoreML/DirectML/OpenVINO)
  - Ensure the shared libraries are discoverable (`DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH`/`PATH`)
  - Compile this crate with the corresponding feature (e.g. `--features cuda`)
- The engine currently stubs image generation while the SDXL ONNX pipeline is wired in. Replace the placeholder in `src/infer/sdxl.rs` with real model session creation and execution.

## Where to extend

- Replace the placeholder generator in `src/infer/sdxl.rs` with:
  - Text encoders, UNet, VAE ONNX sessions
  - Provider selection per device and per session
  - Scheduler and latent updates
  - Multi-GPU sharding by creating one engine per device
- Implement `POST /api/upscale` using ONNX models (e.g., Real-ESRGAN or ESRGAN variants).
