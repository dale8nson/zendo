use axum::{extract::State, response::IntoResponse, Json};
use serde::Serialize;

use crate::{AppState, infer::sdxl::SdxlStatus};

#[derive(Serialize)]
pub struct StatusResponse {
    ready: bool,
    model_dir: Option<String>,
    detail: Option<SdxlStatus>,
    message: Option<String>,
}

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    match state.sdxl_pool.next_engine().await {
        Some(engine) => {
            let st = engine.status();
            Json(StatusResponse { ready: st.ready, model_dir: st.model_dir.clone(), detail: Some(st), message: None })
        }
        None => Json(StatusResponse { ready: false, model_dir: state.cfg.sdxl_model_dir.clone(), detail: None, message: Some("engine not initialized".into()) }),
    }
}

