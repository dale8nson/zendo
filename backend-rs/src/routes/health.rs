use axum::{response::IntoResponse, Json};
use serde::Serialize;

#[derive(Serialize)]
struct Health { status: &'static str }

pub async fn health() -> impl IntoResponse {
    Json(Health { status: "ok" })
}

