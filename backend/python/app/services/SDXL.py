from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
import torch
from pydantic.main import BaseModel
from PIL import Image
import os
import base64
from io import BytesIO
from typing import cast



device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class GenerateRequest(BaseModel):
    prompt: str

pipe = None

async def init_SDXL():
    print("Initializing SDXL pipeline...")
    global pipe, device
    print("Device:", device)
    pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(os.getcwd(),"../models/sdxl-base-1.0/"), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True)
    pipe.safety_checker = None

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.to(device)
    # pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

async def generate(prompt) -> dict:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    global pipe
    if pipe is None:
        print("SDXL pipeline not initialized")
        return {}
    image = cast(Image.Image, pipe(prompt, negative_prompt="text, low quality, stylised, distorted facial features, unnatural, unrealistic, distorted teeth", num_inference_steps=50, guidance_scale=10, width=512, height=512).images[0])
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": image_base64}
