use anyhow::{Error, Result};
use image::{
    imageops::{crop, resize, Lanczos3},
    DynamicImage, GenericImageView, ImageFormat, ImageReader,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};
use std::io::Cursor;
use std::path::Path;

fn resize_image_bytes(data: &[u8], max_dim: u32) -> Result<Vec<u8>> {
    let img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;

    let (w, h) = img.dimensions();
    let scale = max_dim as f32 / w.max(h) as f32;
    let resized = img.resize(
        (w as f32 * scale).round() as u32,
        (h as f32 * scale).round() as u32,
        Lanczos3,
    );

    let mut buf = Vec::new();
    resized.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
    Ok(buf)
}

#[pyfunction(name = "resize_image_bytes")]
fn resize_image_bytes_py(py: Python<'_>, data: &[u8], max_dim: u32) -> PyResult<PyObject> {
    let out =
        resize_image_bytes(data, max_dim).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out).into())
}

fn resize_image<P: AsRef<Path>>(in_path: P, out_path: P, max_dim: u32) -> Result<()> {
    let img = image::open(&in_path)?;
    let (w, h) = img.dimensions();
    let scale = max_dim as f32 / w.max(h) as f32;
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let resized = resize(&img, new_w, new_h, Lanczos3);
    resized.save_with_format(out_path, ImageFormat::Png)?;
    Ok(())
}

#[pyfunction(name = "resize_image")]
fn resize_image_py(_py: Python, inp: &str, outp: &str, max_dim: u32) -> PyResult<()> {
    resize_image(inp, outp, max_dim)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

fn crop_image(data: &[u8], x: u32, y: u32, w: u32, h: u32) -> Result<Vec<u8>> {
    let mut img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let cropped_img = img.crop(x, y, w, h);
    let mut buf = Vec::new();
    cropped_img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
    Ok(buf)
}

#[pyfunction(name = "crop_image")]
fn crop_image_py(
    py: Python<'_>,
    data: &[u8],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> PyResult<PyObject> {
    let out = crop_image(&data, x, y, w, h).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out).into())
}

#[pymodule(name = "zendolib")]
fn zendolib_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image_py, m)?)?;
    Ok(())
}
