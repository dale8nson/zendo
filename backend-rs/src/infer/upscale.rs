use std::{path::Path, sync::Arc};

use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, Array4, Axis, IxDyn};
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::Value;

use crate::config::Config;

fn normalize_image_to_nchw(img: &DynamicImage) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut arr = Array::zeros((1usize, 3usize, h as usize, w as usize));
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = rgb.get_pixel(x as u32, y as u32);
            arr[[0, 0, y, x]] = (p[0] as f32) / 255.0;
            arr[[0, 1, y, x]] = (p[1] as f32) / 255.0;
            arr[[0, 2, y, x]] = (p[2] as f32) / 255.0;
        }
    }
    arr
}

fn nchw_to_image(arr: &Array4<f32>) -> Result<RgbImage> {
    let shape = arr.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if n != 1 || c != 3 { return Err(anyhow!("unexpected output shape {:?}", shape)); }
    let mut img: RgbImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h { for x in 0..w {
        let r = (arr[[0,0,y,x]].clamp(0.0,1.0) * 255.0) as u8;
        let g = (arr[[0,1,y,x]].clamp(0.0,1.0) * 255.0) as u8;
        let b = (arr[[0,2,y,x]].clamp(0.0,1.0) * 255.0) as u8;
        img.put_pixel(x as u32, y as u32, Rgb([r,g,b]));
    }}
    Ok(img)
}

pub struct UpscaleEngine {
    session: tokio::sync::Mutex<Session>,
}

impl UpscaleEngine {
    pub async fn new<P: AsRef<Path>>(onnx_path: P, _cfg: Arc<Config>) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(onnx_path.as_ref())
            .with_context(|| format!("failed to load upscaler onnx at {:?}", onnx_path.as_ref()))?;

        Ok(Self { session: tokio::sync::Mutex::new(session) })
    }

    pub async fn upscale_rgb_png_bytes(&self, png_bytes: &[u8]) -> Result<Vec<u8>> {
        let img = image::load_from_memory(png_bytes)?;
        self.upscale_image(img).await
    }

    pub async fn upscale_image(&self, img: DynamicImage) -> Result<Vec<u8>> {
        let input = normalize_image_to_nchw(&img);
        let input_shape: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let input_tensor = Value::from_array(input)?;

        let mut session = self.session.lock().await;
        let outputs = session.run(ort::inputs! { "input" => input_tensor })?;
        // Take the first output
        let out0 = &outputs[0];
        // Downcast to tensor ref of f32 and extract into ndarray view
        let out_ref: ort::value::TensorRef<'_, f32> = out0.view().downcast()?;
        let out_arr_view = out_ref.extract_array();
        let out4 = out_arr_view.to_owned().into_dimensionality::<ndarray::Ix4>()?;
        let up_img = nchw_to_image(&out4)?;
        let mut buf = Vec::new();
        {
            let mut c = std::io::Cursor::new(&mut buf);
            up_img.write_to(&mut c, image::ImageFormat::Png)?;
        }
        Ok(buf)
    }
}
