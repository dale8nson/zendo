from diffusers import StableDiffusionXLPipeline,StableDiffusionXLInpaintPipeline, AutoPipelineForInpainting
import os
import torch

if os.path.exists("sdxl-base-1.0"):
    base = StableDiffusionXLPipeline.from_pretrained("sdxl-base-1.0")

else:
    raise ValueError("Base model not found. Please run `python backend/models/setup_sdxl.py first.`")

inpainter = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
inpainter.config.requires_aesthetics_score = False
if hasattr(inpainter, "register_to_config"):
    inpainter.register_to_config(requires_aesthetics_score=False)

inpainter.save_pretrained("stable-diffusion-xl-inpainting-1.0", max_shard_size="5GB")
