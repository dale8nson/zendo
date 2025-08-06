from diffusers import StableDiffusionUpscalePipeline
import torch

upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True
)

upscaler.save_pretrained("stable-diffusion-x4-upscaler")
