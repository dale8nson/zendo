from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import os
import torch

if os.path.exists("sdxl-base-1.0"):
    base = StableDiffusionXLPipeline.from_pretrained("sdxl-base-1.0")

else:
    raise ValueError("Base model not found. Please run `python backend/models/setup_sdxl.py first.`")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
refiner.config.requires_aesthetics_score = False
if hasattr(refiner, "register_to_config"):
    refiner.register_to_config(requires_aesthetics_score=False)

refiner.save_pretrained("sdxl-refiner-1.0", max_shard_size="5GB")
