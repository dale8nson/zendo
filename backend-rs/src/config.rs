use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub sdxl_model_dir: Option<String>,
    pub concurrency: usize,
    pub steps_default: u32,
    pub upscaler_onnx: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        let bind_addr = env::var("ZENDO_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        let sdxl_model_dir = env::var("ZENDO_SDXL_DIR").ok();
        let concurrency = env::var("ZENDO_CONCURRENCY").ok().and_then(|s| s.parse().ok()).unwrap_or_else(num_cpus::get);
        let steps_default = env::var("ZENDO_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
        let upscaler_onnx = env::var("ZENDO_UPSCALER_ONNX").ok();
        Self { bind_addr, sdxl_model_dir, concurrency, steps_default, upscaler_onnx }
    }
}
