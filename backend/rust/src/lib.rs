use anyhow::{Error, Result};
use fast_image_resize as fir;
use fir::{images::Image, PixelType, ResizeOptions, Resizer};

use image::{
    imageops::{crop, resize, FilterType, Lanczos3},
    DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageReader, RgbImage,
};

use std::io::Cursor;
use std::path::Path;

pub mod infer;

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

pub fn transform_image(path: &str, scale: f32) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    let mut src_image = Image::from_vec_u8(width, height, rgb.into_raw(), PixelType::U8x3)?;
    let mut dst_image = Image::new(new_width, new_height, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new();

    resizer
        .resize(&mut src_image, &mut dst_image, Some(&options))
        .expect("Resize failed");

    let buffer = ImageBuffer::from_raw(new_width, new_height, dst_image.buffer().to_vec())
        .ok_or("Failed to create image buffer")?;

    Ok(DynamicImage::ImageRgb8(buffer))
}
