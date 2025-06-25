from diffusers import StableDiffusionXLPipeline
import torch
from pydantic.main import BaseModel
from PIL import Image
import os
import base64
from io import BytesIO
from typing import cast



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class GenerateRequest(BaseModel):
    prompt: str

pipe = None

async def init_SDXL():
    global pipe, device
    pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(os.getcwd(),"../models/sdxl-base-1.0/"), torch_dtype=torch.float32, use_safetensors=True).to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

async def generate(prompt) -> dict:
    global pipe
    if pipe is None:
        print("SDXL pipeline not initialized")
        return {}
    image = cast(Image.Image, pipe(prompt, num_inference_steps=10,   # <- this makes it much faster
        guidance_scale=5.0, width=512, height=512).images[0])
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": image_base64}
