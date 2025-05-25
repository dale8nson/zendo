use anyhow::{Error, Result};
use image::{
    imageops::{crop, resize, FilterType, Lanczos3},
    DynamicImage, GenericImageView, ImageFormat, ImageReader,
};

use std::io::Cursor;
use std::path::Path;

#[cfg(feature = "python")]
pub mod python;

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

fn crop_image(data: &[u8], x: u32, y: u32, w: u32, h: u32) -> Result<Vec<u8>> {
    let mut img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let cropped_img = img.crop(x, y, w, h);
    let mut buf = Vec::new();
    cropped_img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
    Ok(buf)
}

pub fn transform_image(path: &str, scale: f32) -> Result<DynamicImage, image::ImageError> {
    let img = image::open(path)?;

    let (width, height) = img.dimensions();
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    let scaled = img.resize(new_width, new_height, FilterType::CatmullRom);

    Ok(scaled)
}
