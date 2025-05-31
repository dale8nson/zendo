use anyhow::Result;
use image::{imageops::FilterType, GenericImageView};
use tch::{CModule, Device, Kind, Tensor};

pub struct ClipModel {
    model: CModule,
    device: Device,
}

impl ClipModel {
    pub fn new(model_path: &str) -> Result<Self> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(model_path, device)?;
        Ok(Self { model, device })
    }

    pub fn encode_image(&self, image_tensor: &Tensor) -> Result<Tensor> {
        let image_tensor = image_tensor.to_device(self.device);
        let features = self.model.forward_ts(&[image_tensor])?;
        Ok(features)
    }
}

pub fn preprocess_image(path: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = img.dimensions();

    let size = w.min(h);
    let cropped =
        image::imageops::crop_imm(&img, (w - size) / 2, (h - size) / 2, size, size).to_image();

    let resized = image::imageops::resize(&cropped, 224, 224, FilterType::CatmullRom);
    let rgb_flat: Vec<f32> = resized
        .pixels()
        .flat_map(|p| p.0.iter().map(|&x| x as f32 / 255.0))
        .collect();

    let tensor = Tensor::f_from_slice(&rgb_flat)?
        .to_kind(Kind::Float)
        .reshape(&[1, 224, 224, 3])
        .permute(&[0, 3, 1, 2]);

    let mean = &Tensor::f_from_slice(&[0.48145466, 0.4578275, 0.40821073])?
        .to_kind(Kind::Float)
        .view([1, 3, 1, 1]);
    let std = &Tensor::f_from_slice(&[0.26862954, 0.26130258, 0.27577711])?
        .to_kind(Kind::Float)
        .view([1, 3, 1, 1]);

    let normalized = (tensor - mean) / std;

    Ok(normalized.squeeze())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_clip_model_encode_image() {
        let model_path = Path::new("../models/openclip/ViT-B-32-scripted.pt");
        let image_path =
            Path::new("../python/app/uploads/6e0fbd28-e451-4dc8-90b5-6fc25b25c106.webp");

        let model = ClipModel::new(model_path.to_str().unwrap()).expect("Failed to load model");

        let image_tensor = super::preprocess_image(image_path.to_str().unwrap())
            .expect("Failed to preprocess image");

        let features = model
            .encode_image(&image_tensor)
            .expect("Failed to encode image");

        println!("Encoded image shape: {:?}", features.size());

        assert_eq!(features.size()[1], 512);
    }
}
