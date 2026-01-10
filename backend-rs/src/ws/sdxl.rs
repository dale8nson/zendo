use std::sync::Arc;

use axum::{
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}, connect_info::ConnectInfo},
    response::IntoResponse,
    Json,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::net::SocketAddr;
use tracing::{info, warn};

use crate::{infer::sdxl::{SdxlEngineEvent, SdxlParams}, AppState};

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WsClientMsg {
    Generate { request_id: Option<Uuid>, params: SdxlParams },
    Ping,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WsServerMsg {
    Ready,
    Progress { request_id: Uuid, step: u32, total: u32 },
    Result { request_id: Uuid, image_base64: String, seed: u64, duration_ms: u128 },
    Error { request_id: Option<Uuid>, message: String },
    Pong,
}

pub async fn ws_handler(
    State(state): State<crate::AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    info!(%addr, "ws /ws/sdxl upgrade request");
    ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: AppState, mut socket: WebSocket, addr: SocketAddr) {
    info!(%addr, "ws /ws/sdxl connected");
    let _ = socket.send(Message::Text(serde_json::to_string(&WsServerMsg::Ready).unwrap())).await;

    while let Some(Ok(msg)) = socket.next().await {
        match msg {
            Message::Text(text) => {
                match serde_json::from_str::<WsClientMsg>(&text) {
                    Ok(WsClientMsg::Ping) => {
                        info!(%addr, "ws /ws/sdxl ping");
                        let _ = socket.send(Message::Text(serde_json::to_string(&WsServerMsg::Pong).unwrap())).await;
                    }
                    Ok(WsClientMsg::Generate { request_id, params }) => {
                        let req_id = request_id.unwrap_or_else(Uuid::new_v4);
                        info!(%addr, %req_id, prompt=%params.prompt, steps=?params.steps, width=params.width, height=params.height, "ws /ws/sdxl generate received");
                        // Early progress ping so client sees we're working while models load
                        let _ = socket.send(Message::Text(
                            serde_json::to_string(&WsServerMsg::Progress { request_id: req_id, step: 0, total: params.steps.unwrap_or(25) }).unwrap()
                        )).await;
                        match state.sdxl_pool.next_engine().await {
                            Some(engine) => {
                                let mut event_stream = engine.generate_streaming(params).await;

                                while let Some(evt) = event_stream.next().await {
                                    match evt {
                                        SdxlEngineEvent::Progress { step, total } => {
                                            if step % 5 == 0 || step == total { info!(%addr, %req_id, step, total, "ws /ws/sdxl progress"); }
                                            let _ = socket.send(Message::Text(
                                                serde_json::to_string(&WsServerMsg::Progress { request_id: req_id, step, total }).unwrap()
                                            )).await;
                                        }
                                        SdxlEngineEvent::Result { image_base64, seed, duration_ms } => {
                                            info!(%addr, %req_id, duration_ms, seed, "ws /ws/sdxl result ready");
                                            let _ = socket.send(Message::Text(
                                                serde_json::to_string(&WsServerMsg::Result { request_id: req_id, image_base64, seed, duration_ms }).unwrap()
                                            )).await;
                                            break;
                                        }
                                        SdxlEngineEvent::Error { message } => {
                                            warn!(%addr, %req_id, %message, "ws /ws/sdxl error");
                                            let _ = socket.send(Message::Text(
                                                serde_json::to_string(&WsServerMsg::Error { request_id: Some(req_id), message }).unwrap()
                                            )).await;
                                            break;
                                        }
                                    }
                                }
                            }
                            None => {
                                let _ = socket.send(Message::Text(
                                    serde_json::to_string(&WsServerMsg::Error { request_id: Some(req_id), message: "SDXL engine initialization failed".into() }).unwrap()
                                )).await;
                            }
                        }
                    }
                    Err(e) => {
                        warn!(%addr, error=%e.to_string(), "ws /ws/sdxl parse error");
                        let _ = socket.send(Message::Text(serde_json::to_string(&WsServerMsg::Error { request_id: None, message: format!("parse error: {}", e) }).unwrap())).await;
                    }
                }
            }
            Message::Binary(_) => {}
            Message::Ping(_) => { let _ = socket.send(Message::Pong(vec![])).await; }
            Message::Pong(_) => {}
            Message::Close(_) => { info!(%addr, "ws /ws/sdxl closed by client"); break },
        }
    }
    info!(%addr, "ws /ws/sdxl connection ended");
}
