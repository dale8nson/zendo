use anyhow::Result;
use image::{imageops, DynamicImage, GenericImageView, ImageFormat};
use pyo3::prelude::*;
use std::path::Path;

pub fn resize_image<P: AsRef<Path>>(in_path: P, out_path: P, max_dim: u32) -> Result<()> {
    let img = image::open(&in_path)?;
    let (w, h) = img.dimensions();
    let scale = max_dim as f32 / w.max(h) as f32;
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let resized = imageops::resize(&img, new_w, new_h, imageops::Lanczos3);
    resized.save_with_format(out_path, ImageFormat::Png)?;
    Ok(())
}

#[pyfunction(name = "resize_image")]
fn resize_image_py(_py: Python, inp: &str, outp: &str, max_dim: u32) -> PyResult<()> {
    resize_image(inp, outp, max_dim)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// pub fn crop_image(img: &mut DynamicImage, x: u32, y: u32, w: u32, h: u32) -> Result<DynamicImage> {

// }

#[pymodule(name = "zendolib")]
fn zendolib_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image_py, m)?)?;
    Ok(())
}
