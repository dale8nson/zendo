use std::{sync::{atomic::{AtomicUsize, Ordering}, Arc}, time::Instant};

use anyhow::{anyhow, Context, Result};
use futures::{Stream, StreamExt};
use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, time::{sleep, Duration}};

use crate::config::Config;
use ndarray::{Array, Array0, Array1, Array2, Array3, Array4, Axis, Ix4};
use ort::{
    session::{builder::{GraphOptimizationLevel, SessionBuilder}, Session},
    value::Value,
};
use base64::Engine;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

fn resolve_dir_path(dir: &str) -> Option<String> {
    let p = Path::new(dir);
    if p.exists() { return Some(dir.to_string()); }
    let candidates = [
        Path::new(".").join(dir),
        Path::new("..").join(dir),
        Path::new("../..").join(dir),
    ];
    for c in candidates.iter() {
        if c.exists() { return Some(c.to_string_lossy().into_owned()); }
    }
    None
}

fn try_load_tokenizer_from_json(path: &Path) -> Option<tokenizers::Tokenizer> {
    if path.exists() {
        tokenizers::Tokenizer::from_file(path.to_string_lossy().as_ref()).ok()
    } else { None }
}

fn build_bpe_tokenizer_from_vocab_merges(vocab: &Path, merges: &Path) -> Option<tokenizers::Tokenizer> {
    use tokenizers::{models::bpe::BPE, Tokenizer};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::decoders::byte_level::ByteLevel as ByteDecoder;
    let bpe = BPE::from_file(
        &vocab.to_string_lossy(),
        &merges.to_string_lossy(),
    ).build().ok()?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(ByteLevel::default());
    tok.with_decoder(ByteDecoder::default());
    Some(tok)
}

fn try_build_tokenizer_from_dir(base: &Path) -> Option<tokenizers::Tokenizer> {
    // Prefer tokenizer.json; fallback to vocab.json + merges.txt
    let json = base.join("tokenizer.json");
    if let Some(t) = try_load_tokenizer_from_json(&json) { return Some(t); }
    let vocab = base.join("vocab.json");
    let merges = base.join("merges.txt");
    build_bpe_tokenizer_from_vocab_merges(&vocab, &merges)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SdxlParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
}

impl Default for SdxlParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            seed: None,
            width: 1024,
            height: 1024,
            steps: None,
            guidance_scale: Some(7.5),
        }
    }
}

#[derive(Debug)]
pub enum SdxlEngineEvent {
    Progress { step: u32, total: u32 },
    Result { image_base64: String, seed: u64, duration_ms: u128 },
    Error { message: String },
}

pub struct SdxlOnnxEngine {
    device_label: String,
    model_dir: Option<String>,
    te1: Option<Arc<tokio::sync::Mutex<Session>>>,
    te2: Option<Arc<tokio::sync::Mutex<Session>>>,
    unet: Option<Arc<tokio::sync::Mutex<Session>>>,
    vae_dec: Option<Arc<tokio::sync::Mutex<Session>>>,
    tok1: Option<Arc<tokenizers::Tokenizer>>,
    tok2: Option<Arc<tokenizers::Tokenizer>>,
}

#[derive(Serialize)]
pub struct SdxlStatus {
    pub ready: bool,
    pub model_dir: Option<String>,
    pub te1: bool,
    pub te2: bool,
    pub unet: bool,
    pub vae_decoder: bool,
    pub tokenizer1: bool,
    pub tokenizer2: bool,
    pub device: String,
}

impl SdxlOnnxEngine {
    pub fn status(&self) -> SdxlStatus {
        SdxlStatus {
            ready: self.te1.is_some() && self.te2.is_some() && self.unet.is_some() && self.vae_dec.is_some(),
            model_dir: self.model_dir.clone(),
            te1: self.te1.is_some(),
            te2: self.te2.is_some(),
            unet: self.unet.is_some(),
            vae_decoder: self.vae_dec.is_some(),
            tokenizer1: self.tok1.is_some(),
            tokenizer2: self.tok2.is_some(),
            device: self.device_label.clone(),
        }
    }
}

impl SdxlOnnxEngine {
    pub async fn new(device_label: String, model_dir: Option<String>) -> Result<Self> {
        // Try to load sessions and tokenizers if available (best-effort)
        let mut te1 = None;
        let mut te2 = None;
        let mut unet = None;
        let mut vae_dec = None;
        let mut tok1 = None;
        let mut tok2 = None;
        if let Some(dir) = &model_dir {
            let resolved = resolve_dir_path(dir).unwrap_or_else(|| dir.clone());
            let dir = resolved;
            info!(model_dir=%dir, "Initializing SDXL ONNX engine");
            // Heuristics: if dir points to an ONNX export, expect text_encoder/model.onnx
            let is_coreml = device_label.contains("coreml");
            let opt_level_for = |is_coreml: bool| if is_coreml { GraphOptimizationLevel::Level1 } else { GraphOptimizationLevel::Level3 };
            let te_path = [
                format!("{}/text_encoder/model.onnx", dir),
                format!("{}/text_encoder_1/model.onnx", dir),
            ]
            .into_iter()
            .find(|p| std::path::Path::new(p).exists());

            if let Some(p) = te_path {
                info!(path=%p, device=%device_label, "Loading TE1 session");
                let sess = SessionBuilder::new()?
                    .with_optimization_level(opt_level_for(is_coreml))?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_file(&p)
                    .with_context(|| format!("failed to load TE1 from {}", p))?;
                te1 = Some(Arc::new(tokio::sync::Mutex::new(sess)));
            } else {
                warn!("TE1 ONNX not found under {}/text_encoder", dir);
            }

            let te2_path = [format!("{}/text_encoder_2/model.onnx", dir)]
                .into_iter()
                .find(|p| std::path::Path::new(p).exists());
            if let Some(p) = te2_path {
                info!(path=%p, device=%device_label, "Loading TE2 session");
                // If this ONNX relies on external data, ensure the sibling file exists to avoid
                // opaque loader errors like ".../model.onnx/model.onnx_data: Not a directory".
                let ext_data = Path::new(&p).with_file_name("model.onnx_data");
                if !ext_data.exists() {
                    return Err(anyhow!(
                        "TE2 external data missing: expected {}. Restore the export (model.onnx + model.onnx_data) or point ZENDO_SDXL_DIR to a complete ONNX export.",
                        ext_data.to_string_lossy()
                    ));
                }
                let sess = SessionBuilder::new()?
                    .with_optimization_level(opt_level_for(is_coreml))?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_file(&p)
                    .with_context(|| format!("failed to load TE2 from {}", p))?;
                te2 = Some(Arc::new(tokio::sync::Mutex::new(sess)));
            }

            let unet_path = format!("{}/unet/model.onnx", dir);
            if std::path::Path::new(&unet_path).exists() {
                info!(path=%unet_path, device=%device_label, "Loading UNet session");
                let sess = SessionBuilder::new()?
                    .with_optimization_level(opt_level_for(is_coreml))?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_file(&unet_path)
                    .with_context(|| format!("failed to load UNet from {}", unet_path))?;
                unet = Some(Arc::new(tokio::sync::Mutex::new(sess)));
            }

            let vae_path = format!("{}/vae_decoder/model.onnx", dir);
            if std::path::Path::new(&vae_path).exists() {
                info!(path=%vae_path, device=%device_label, "Loading VAE decoder session");
                let sess = SessionBuilder::new()?
                    .with_optimization_level(opt_level_for(is_coreml))?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_file(&vae_path)
                    .with_context(|| format!("failed to load VAE decoder from {}", vae_path))?;
                vae_dec = Some(Arc::new(tokio::sync::Mutex::new(sess)));
            }

            // Tokenizer: try tokenizer.json, else vocab+merges
            if tok1.is_none() {
                let base = Path::new(&dir).join("tokenizer");
                if let Some(tok) = try_build_tokenizer_from_dir(&base) { tok1 = Some(Arc::new(tok)); }
            }
            if tok2.is_none() {
                let base = Path::new(&dir).join("tokenizer_2");
                if let Some(tok) = try_build_tokenizer_from_dir(&base) { tok2 = Some(Arc::new(tok)); }
            }
            if tok1.is_none() || tok2.is_none() {
                // Try sibling of ONNX dir (e.g., parent tokenizer folders)
                if let Some(parent) = Path::new(&dir).parent() {
                    if tok1.is_none() {
                        let base = parent.join("tokenizer");
                        if let Some(tok) = try_build_tokenizer_from_dir(&base) { tok1 = Some(Arc::new(tok)); }
                    }
                    if tok2.is_none() {
                        let base = parent.join("tokenizer_2");
                        if let Some(tok) = try_build_tokenizer_from_dir(&base) { tok2 = Some(Arc::new(tok)); }
                    }
                }
            }
        }

        // Require all core sessions to be present
        if te1.is_none() { return Err(anyhow!("TE1 not found or failed to load")); }
        if te2.is_none() { return Err(anyhow!("TE2 not found or failed to load")); }
        if unet.is_none() { return Err(anyhow!("UNet not found or failed to load")); }
        if vae_dec.is_none() { return Err(anyhow!("VAE decoder not found or failed to load")); }

        Ok(Self { device_label, model_dir, te1, te2, unet, vae_dec, tok1, tok2 })
    }

    pub async fn generate_streaming(&self, params: SdxlParams) -> impl Stream<Item = SdxlEngineEvent> {
        let (tx, rx) = mpsc::unbounded_channel::<SdxlEngineEvent>();
        let steps = params.steps.unwrap_or(25);
        let width = params.width.max(128).min(2048);
        let height = params.height.max(128).min(2048);
        let seed = params.seed.unwrap_or_else(|| fastrand::u64(..));
        let prompt = params.prompt.clone();
        let neg_prompt = params.negative_prompt.clone();
        let guidance = params.guidance_scale.unwrap_or(7.5).max(0.25);

        let te1_for_task = self.te1.clone();
        let te2_for_task = self.te2.clone();
        let unet_for_task = self.unet.clone();
        let vae_for_task = self.vae_dec.clone();
        let tok1_for_task = self.tok1.clone();
        let tok2_for_task = self.tok2.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            let pipeline_res: Result<()> = (|| -> Result<()> {
                let (te1, te2, unet, vae, tok1, tok2) = (
                    te1_for_task.as_ref().context("TE1 not loaded")?,
                    te2_for_task.as_ref().context("TE2 not loaded")?,
                    unet_for_task.as_ref().context("UNet not loaded")?,
                    vae_for_task.as_ref().context("VAE decoder not loaded")?,
                    tok1_for_task.as_ref().context("tokenizer 1 not loaded")?,
                    tok2_for_task.as_ref().context("tokenizer 2 not loaded")?,
                );

                // Tokenize
                let seq_len = 77usize;
                const CLIP_BOS_ID: i64 = 49406;
                const CLIP_EOS_ID: i64 = 49407;
                let encode = |tok: &tokenizers::Tokenizer, s: &str| -> Result<Vec<i64>> {
                    let e = tok.encode(s, false).map_err(|e| anyhow!(e))?;
                    let mut ids: Vec<i64> = e.get_ids().iter().map(|&x| x as i64).collect();
                    // Add BOS/EOS explicitly
                    let mut with_special = Vec::with_capacity(ids.len()+2);
                    with_special.push(CLIP_BOS_ID);
                    with_special.extend(ids.drain(..));
                    with_special.push(CLIP_EOS_ID);
                    let mut ids = with_special;
                    ids.truncate(seq_len);
                    if ids.len() < seq_len { ids.resize(seq_len, 0); }
                    Ok(ids)
                };
                let ids1_pos = encode(tok1, &prompt)?;
                let ids1_neg = encode(tok1, &neg_prompt)?;
                let ids2_pos = encode(tok2, &prompt)?;
                let ids2_neg = encode(tok2, &neg_prompt)?;

                // Run TE1+TE2 to get hidden states and text_embeds
                let run_te = |sess: &Arc<tokio::sync::Mutex<Session>>, ids: &Vec<i64>| -> Result<(Array3<f32>, Option<Array2<f32>>)> {
                    let a = Array2::from_shape_vec((1, seq_len), ids.clone())?;
                    let ids_val = Value::from_array(a)?;
                    let mut guard = futures::executor::block_on(sess.lock());
                    let outs = guard.run(ort::inputs!{ "input_ids" => ids_val })?;
                    let hs = outs["last_hidden_state"].view().downcast::<ort::value::TensorValueType<f32>>()?;
                    let hs_arr = hs.extract_array().to_owned().into_dimensionality::<ndarray::Ix3>()?;
                    let te = outs.get("text_embeds").and_then(|_| Some(outs["text_embeds"].view().downcast::<ort::value::TensorValueType<f32>>().ok())).flatten()
                        .map(|v| v.extract_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap());
                    Ok((hs_arr, te))
                };
                let (hs1_pos, _) = run_te(te1, &ids1_pos)?;
                let (hs1_neg, _) = run_te(te1, &ids1_neg)?;
                let (hs2_pos, te2_pos_opt) = run_te(te2, &ids2_pos)?;
                let (hs2_neg, te2_neg_opt) = run_te(te2, &ids2_neg)?;
                let te2_pool_pos = te2_pos_opt.context("text_embeds missing in TE2 outputs")?;
                let te2_pool_neg = te2_neg_opt.context("text_embeds missing in TE2 outputs (neg)")?;

                // Concat hidden states to 2048 dim
                let enc_pos = ndarray::concatenate(Axis(2), &[hs1_pos.view(), hs2_pos.view()])?;
                let enc_neg = ndarray::concatenate(Axis(2), &[hs1_neg.view(), hs2_neg.view()])?;

                // Time ids [1,6]
                let (th, tw) = (height as f32, width as f32);
                let time_ids: Array2<f32> = Array2::from_shape_vec((1,6), vec![th, tw, 0.0, 0.0, th, tw])?;

                // Init latents
                let (lh, lw) = (height/8, width/8);
                let mut latents = Array4::<f32>::zeros((1,4, lh as usize, lw as usize));
                let mut rng = fastrand::Rng::with_seed(seed);
                for v in latents.iter_mut() { *v = rng.f32() * 2.0 - 1.0; }

                // DDIM-like schedule
                let num_train = 1000usize;
                let beta_start = 0.00085f32;
                let beta_end = 0.012f32;
                // scaled_linear as in diffusers
                let bstart = beta_start.sqrt();
                let bend = beta_end.sqrt();
                let mut betas = Vec::with_capacity(num_train);
                for i in 0..num_train {
                    let r = i as f32 / (num_train as f32 - 1.0);
                    let v = bstart + (bend - bstart) * r;
                    betas.push(v * v);
                }
                let mut alphas = Vec::with_capacity(num_train);
                for b in &betas { alphas.push(1.0 - *b); }
                let mut alpha_cumprod = Vec::with_capacity(num_train);
                let mut cur = 1.0f32;
                for a in &alphas { cur *= *a; alpha_cumprod.push(cur); }

                let t_indices: Vec<usize> = (0..steps).map(|i| {
                    let a = i as f32 / (steps as f32 - 1.0);
                    let t = ((num_train - 1) as f32 * (1.0 - a)).round() as isize;
                    t.max(0) as usize
                }).collect();

                let run_unet = |sess: &Arc<tokio::sync::Mutex<Session>>, lat: &Array4<f32>, enc: &Array3<f32>, text_emb: &Array2<f32>, time_ids: &Array2<f32>, t: f32| -> Result<Array4<f32>> {
                    let sample = Value::from_array(lat.clone())?;
                    let timestep = Value::from_array(Array0::from_elem((), t))?;
                    let encv = Value::from_array(enc.clone())?;
                    let tev = Value::from_array(text_emb.clone())?;
                    let tiv = Value::from_array(time_ids.clone())?;
                    let mut guard = futures::executor::block_on(sess.lock());
                    let outs = guard.run(ort::inputs!{
                        "sample" => sample,
                        "timestep" => timestep,
                        "encoder_hidden_states" => encv,
                        "text_embeds" => tev,
                        "time_ids" => tiv,
                    })?;
                    let out = &outs[0];
                    let arr = out.view().downcast::<ort::value::TensorValueType<f32>>()?.extract_array().to_owned();
                    Ok(arr.into_dimensionality::<Ix4>()?)
                };

                for (i, &t_idx) in t_indices.iter().enumerate() {
                    let _ = tx.send(SdxlEngineEvent::Progress { step: (i+1) as u32, total: steps });
                    let a_t = alpha_cumprod[t_idx].clamp(1e-8, 0.999999);
                    let a_prev = if t_idx > 0 { alpha_cumprod[t_idx - 1].clamp(1e-8, 0.999999) } else { 1.0 };
                    let t_val = t_idx as f32;

                    let eps_uncond = run_unet(unet, &latents, &enc_neg, &te2_pool_neg, &time_ids, t_val)?;
                    let eps_cond = run_unet(unet, &latents, &enc_pos, &te2_pool_pos, &time_ids, t_val)?;
                    let mut eps = eps_uncond.clone();
                    ndarray::Zip::from(eps.view_mut()).and(eps_uncond.view()).and(eps_cond.view()).for_each(|e, u, c| {
                        *e = u + guidance as f32 * (c - u);
                    });

                    // DDIM (eta=0): x0 = (x - sqrt(1-a_t)*eps)/sqrt(a_t); x_prev = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps
                    let sqrt_at = a_t.sqrt();
                    let sqrt_one_minus_at = (1.0 - a_t).sqrt();
                    let sqrt_a_prev = a_prev.sqrt();
                    let sqrt_one_minus_a_prev = (1.0 - a_prev).sqrt();

                    let mut x0 = latents.clone();
                    ndarray::Zip::from(x0.view_mut()).and(eps.view()).for_each(|x0p, e| {
                        *x0p = (*x0p - sqrt_one_minus_at * *e) / sqrt_at;
                    });
                    ndarray::Zip::from(latents.view_mut()).and(x0.view()).and(eps.view()).for_each(|lp, x0p, e| {
                        *lp = sqrt_a_prev * *x0p + sqrt_one_minus_a_prev * *e;
                    });
                }

                // Decode with VAE (scale factor 0.13025)
                let scaled = latents.mapv(|v| v / 0.13025f32);
                let input = Value::from_array(scaled)?;
                let mut guard = futures::executor::block_on(vae.lock());
                let outs = guard.run(ort::inputs!{ "latent_sample" => input })?;
                let out = &outs[0];
                let img = out.view().downcast::<ort::value::TensorValueType<f32>>()?.extract_array().to_owned();
                let img4 = img.into_dimensionality::<Ix4>()?; // [1,3,H,W]
                let img3 = img4.index_axis(Axis(0), 0).to_owned();
                let (c,h,w) = (img3.shape()[0], img3.shape()[1], img3.shape()[2]);
                let mut imgbuf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
                for y in 0..h { for x in 0..w {
                    let r = ((img3[[0,y,x]].clamp(-1.0,1.0) + 1.0) * 0.5 * 255.0) as u8;
                    let g = ((img3[[1,y,x]].clamp(-1.0,1.0) + 1.0) * 0.5 * 255.0) as u8;
                    let b = ((img3[[2,y,x]].clamp(-1.0,1.0) + 1.0) * 0.5 * 255.0) as u8;
                    imgbuf.put_pixel(x as u32, y as u32, Rgb([r,g,b]));
                }}
                let mut buf = Vec::new();
                {
                    let mut c = std::io::Cursor::new(&mut buf);
                    imgbuf.write_to(&mut c, image::ImageFormat::Png)?;
                }
                let b64 = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf));
                let _ = tx.send(SdxlEngineEvent::Result { image_base64: b64, seed, duration_ms: start.elapsed().as_millis() });
                Ok(())
            })();
            if let Err(e) = pipeline_res {
                let _ = tx.send(SdxlEngineEvent::Error { message: format!("sdxl pipeline error: {}", e) });
            }
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
}

pub struct SdxlEnginePool {
    inner: tokio::sync::OnceCell<Vec<Arc<SdxlOnnxEngine>>>,
    cfg: Arc<Config>,
    rr: AtomicUsize,
}

impl SdxlEnginePool {
    pub fn new_lazy(cfg: Arc<Config>) -> Self {
        Self { inner: tokio::sync::OnceCell::new(), cfg, rr: AtomicUsize::new(0) }
    }

    async fn init_engines(&self) -> Result<Vec<Arc<SdxlOnnxEngine>>> {
        let mut engines: Vec<Arc<SdxlOnnxEngine>> = Vec::new();

        // Try CUDA if compiled
        #[cfg(feature = "cuda")]
        match SdxlOnnxEngine::new("cuda:0".into(), self.cfg.sdxl_model_dir.clone()).await {
            Ok(engine) => {
                tracing::info!("Initialized SDXL engine (CUDA)");
                engines.push(Arc::new(engine));
            }
            Err(e) => {
                tracing::warn!(error=?e, "Failed to initialize SDXL CUDA engine; will try other providers");
            }
        }

        // Try CoreML if compiled (and if CUDA didn't already succeed)
        #[cfg(feature = "coreml")]
        if engines.is_empty() {
            match SdxlOnnxEngine::new("coreml".into(), self.cfg.sdxl_model_dir.clone()).await {
                Ok(engine) => {
                    tracing::info!("Initialized SDXL engine (CoreML)");
                    engines.push(Arc::new(engine));
                }
                Err(e) => {
                    tracing::warn!(error=?e, "Failed to initialize SDXL CoreML engine; will try CPU fallback");
                }
            }
        }

        // Always attempt CPU fallback if nothing else worked
        if engines.is_empty() {
            match SdxlOnnxEngine::new("cpu".into(), self.cfg.sdxl_model_dir.clone()).await {
                Ok(engine) => {
                    tracing::info!("Initialized SDXL engine (CPU)");
                    engines.push(Arc::new(engine));
                }
                Err(e) => {
                    tracing::warn!(error=?e, "Failed to initialize SDXL CPU engine");
                }
            }
        }

        if engines.is_empty() {
            Err(anyhow!("no SDXL engines initialized"))
        } else {
            Ok(engines)
        }
    }

    pub async fn next_engine(&self) -> Option<Arc<SdxlOnnxEngine>> {
        let engines = self.inner.get_or_init(|| async {
            // If init fails, keep trying on next call by returning empty vec.
            match self.init_engines().await {
                Ok(v) if !v.is_empty() => v,
                _ => Vec::new(),
            }
        }).await;
        if engines.is_empty() {
            // Fallback: attempt one more time synchronously
            if let Ok(v) = self.init_engines().await { if !v.is_empty() { let _ = self.inner.set(v); } }
        }
        let engines = match self.inner.get() { Some(v) => v, None => return None };
        if engines.is_empty() { return None; }
        let idx = self.rr.fetch_add(1, Ordering::Relaxed) % engines.len();
        engines.get(idx).cloned()
    }
}
