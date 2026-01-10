use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::State,
    http::Method,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::signal;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};
use tracing::{error, info, Level};
use std::path::{Path, PathBuf};

mod config;
mod infer;
mod routes;
mod ws;

use config::Config;
use tracing::warn;
use infer::sdxl::SdxlEnginePool;
use infer::upscale::UpscaleEngine;

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub sdxl_pool: Arc<SdxlEnginePool>,
    pub upscaler: Option<Arc<UpscaleEngine>>, 
}

#[derive(Serialize)]
struct Info {
    name: &'static str,
    version: &'static str,
    build_ts: Option<&'static str>,
    rustc: Option<&'static str>,
    features: &'static str,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load env from current dir and also try parent (project root) for convenience
    dotenvy::dotenv().ok();
    let _ = dotenvy::from_filename("../.env");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_level(true)
        .compact()
        .init();

    let cfg = Arc::new(Config::from_env());

    // Ensure ONNX Runtime dynamic library is found when using `ort` with `load-dynamic`.
    // Prefer `ZENDO_ORT_DYLIB_PATH`, then `ORT_DYLIB_PATH`. Otherwise, fall back to default search.
    {
        use ort::execution_providers::ExecutionProviderDispatch;
        let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();
        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::CUDAExecutionProvider;
            providers.push(CUDAExecutionProvider::default().build());
        }
        #[cfg(feature = "coreml")]
        {
            use ort::execution_providers::CoreMLExecutionProvider;
            providers.push(CoreMLExecutionProvider::default().build());
        }
        #[cfg(feature = "directml")]
        {
            use ort::execution_providers::DirectMLExecutionProvider;
            providers.push(DirectMLExecutionProvider::default().build());
        }
        #[cfg(feature = "openvino")]
        {
            use ort::execution_providers::OpenVINOExecutionProvider;
            providers.push(OpenVINOExecutionProvider::default().build());
        }

        // Try explicit env var first; else auto-detect Python venv's ORT dylib within the repo.
        let explicit = std::env::var("ZENDO_ORT_DYLIB_PATH").or_else(|_| std::env::var("ORT_DYLIB_PATH")).ok();

        fn find_python_ort_dylib() -> Option<String> {
            // Search a few relative bases for a Python venv ORT dylib
            let bases = [
                Path::new("."),
                Path::new(".."),
                Path::new("../.."),
            ];
            for b in bases.iter() {
                let cand = b.join("backend/python/.venv/lib");
                if !cand.exists() { continue; }
                // Walk pythonX.Y/site-packages/onnxruntime/capi
                if let Ok(py_versions) = std::fs::read_dir(&cand) {
                    for pyv in py_versions.flatten() {
                        let sp = pyv.path().join("site-packages/onnxruntime/capi");
                        if !sp.exists() { continue; }
                        if let Ok(files) = std::fs::read_dir(&sp) {
                            for f in files.flatten() {
                                let p = f.path();
                                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                                    let is_dyn = name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll");
                                    let looks_ort = name.starts_with("libonnxruntime") || name.eq_ignore_ascii_case("onnxruntime.dll");
                                    if is_dyn && looks_ort {
                                        return Some(p.to_string_lossy().into_owned());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        }

        let detected = explicit.or_else(find_python_ort_dylib);

        let init_builder = if let Some(lib) = detected {
            info!(path=%lib, "Loading ORT dylib (custom)");
            ort::environment::init_from(lib).with_execution_providers(providers.as_slice())
        } else {
            info!("Loading ORT dylib (system/default)");
            ort::environment::init().with_execution_providers(providers.as_slice())
        };
        match init_builder.commit() {
            Ok(true) => info!("Initialized ORT environment (custom)"),
            Ok(false) => info!("Using existing ORT environment"),
            Err(e) => warn!(error=?e, "Failed to initialize ORT environment; default runtime will be used"),
        }
    }

    // Lazy-init the SDXL pool so the server can start responding immediately.
    let sdxl_pool = Arc::new(SdxlEnginePool::new_lazy(cfg.clone()));
    let default_upscale_path = "backend/models/Real-ESRGAN/RealESRGAN_x4plus.onnx";
    let chosen_upscale = cfg.upscaler_onnx.as_deref().or_else(|| {
        if std::path::Path::new(default_upscale_path).exists() { Some(default_upscale_path) } else { None }
    });
    let upscaler = match chosen_upscale {
        Some(path) => match UpscaleEngine::new(path, cfg.clone()).await {
            Ok(engine) => Some(Arc::new(engine)),
            Err(e) => { tracing::warn!(error=?e, "failed to init upscaler; continuing without it"); None }
        },
        None => None,
    };

    let state = AppState { cfg: cfg.clone(), sdxl_pool: sdxl_pool.clone(), upscaler };

    // Best-effort, non-blocking SDXL prewarm in background
    let prewarm_pool = sdxl_pool.clone();
    tokio::spawn(async move {
        // Attempt once; errors are logged through session creation
        let _ = prewarm_pool.next_engine().await;
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any)
        .max_age(Duration::from_secs(60 * 60));

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/info", get(info_handler))
        .route("/sdxl/status", get(routes::sdxl::status))
        .route("/api/upscale", post(routes::upscale::upscale))
        .route("/ws/sdxl", get(ws::sdxl::ws_handler))
        .with_state(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = cfg.bind_addr.parse()?;
    info!(%addr, "listening (Axum)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let make = app.into_make_service_with_connect_info::<SocketAddr>();
    axum::serve(listener, make)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).expect("failed to install signal handler");
        sigterm.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn info_handler(State(state): State<AppState>) -> impl IntoResponse {
    let features = {
        let mut feats: Vec<&'static str> = Vec::new();
        #[cfg(feature = "cuda")] feats.push("cuda");
        #[cfg(feature = "tensorrt")] feats.push("tensorrt");
        #[cfg(feature = "coreml")] feats.push("coreml");
        #[cfg(feature = "directml")] feats.push("directml");
        #[cfg(feature = "openvino")] feats.push("openvino");
        if feats.is_empty() { feats.push("cpu"); }
        feats.join(",")
    };

    let info = Info {
        name: "zendo-backend",
        version: env!("CARGO_PKG_VERSION"),
        build_ts: option_env!("VERGEN_BUILD_TIMESTAMP"),
        rustc: option_env!("VERGEN_RUSTC_SEMVER"),
        features: Box::leak(features.into_boxed_str()),
    };
    Json(info)
}
