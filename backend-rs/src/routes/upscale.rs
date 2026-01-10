use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use crate::AppState;
use tracing::info;

#[derive(Deserialize)]
pub struct UpscaleRequest {
    pub image_base64: String,
    pub scale: Option<f32>,
}

#[derive(Serialize)]
pub struct UpscaleResponse {
    pub message: String,
    pub image_base64: Option<String>,
}

pub async fn upscale(State(state): State<AppState>, Json(req): Json<UpscaleRequest>) -> impl IntoResponse {
    info!(has_engine=%state.upscaler.is_some(), payload_len=req.image_base64.len(), "POST /api/upscale");
    let Some(engine) = &state.upscaler else {
        let resp = UpscaleResponse { message: "Upscaler not configured. Set ZENDO_UPSCALER_ONNX".into(), image_base64: None };
        return (StatusCode::NOT_IMPLEMENTED, Json(resp));
    };

    let Ok(img_bytes) = B64.decode(
        req.image_base64.split(',').last().unwrap_or(&req.image_base64)
    ) else {
        let resp = UpscaleResponse { message: "Invalid base64 image".into(), image_base64: None };
        return (StatusCode::BAD_REQUEST, Json(resp));
    };

    match engine.upscale_rgb_png_bytes(&img_bytes).await {
        Ok(out) => {
            let data_url = format!("data:image/png;base64,{}", B64.encode(out));
            let resp = UpscaleResponse { message: "ok".into(), image_base64: Some(data_url) };
            (StatusCode::OK, Json(resp))
        }
        Err(e) => {
            let resp = UpscaleResponse { message: format!("upscale error: {}", e), image_base64: None };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(resp))
        }
    }
}
