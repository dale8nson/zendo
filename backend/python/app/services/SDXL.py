from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline

import torch
from pydantic.main import BaseModel
from PIL import Image, ImageFilter, ImageChops, ImageOps
import os
import base64
from io import BytesIO
from typing import cast
import numpy as np
from diffusers.utils import load_image
# from concurrent.futures import ProcessPoolExecutor
from rembg import remove
import math



device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

pipe = None
refiner = None
inpainter = None
selected_image = Image.Image | None
selected_prompt: str | None = None
negative_prompt: str = "text, 'low quality', cropped, art, stylised, 'distorted facial features', 'distorted teeth', 'missing limbs', 'heads cut off', drawing, painting, cartoon, anime, 'picture frame', blurry, 'out of focus', fuzzy, border"
latent = None
prompt_embeds = None
negative_prompt_embeds = None
buffer = BytesIO()
inpaint_image_count = 0
inpaint_mask_count = 0
inpaint_alpha_mask_count = 0

class GenerateRequest(BaseModel):
    prompt: str
    iterations: int
    guidance_scale: float = 7.5
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str

class RefineRequest(BaseModel):
    prompt: str
    image: str
    strength: float
    guidance_scale: float = 7.5
    negative_prompt: str = negative_prompt
    prompt_2: str
    negative_prompt_2: str = "sharp, detailed, 8k, high resolution"


class InpaintRequest(BaseModel):
    image: str
    prompt: str
    mask: str
    strength: float
    guidance_scale: float = 7.5,
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str = "sharp, detailed, 8k, high resolution"
    alpha: int = 255
    noise: float = 0.5
    noise_offset: float = 0.0
    blur: int = 2
    strict: bool = False
    reverse_mask: bool = False

async def to_async(model, dev) -> None:
    model.to(dev)

async def init_inpainter():
    global refiner, inpainter
    print("Initializing SDXL inpainter...")
    inpainter = AutoPipelineForInpainting.from_pipe(cast(StableDiffusionXLImg2ImgPipeline, refiner))
    inpainter.register_to_config(requires_aesthetics_score=False)
    inpainter.safety_checker = None
    inpainter.enable_attention_slicing()
    inpainter.to(device)


async def init_SDXL():
    print("Initializing SDXL pipeline...")
    global pipe, device, refiner, inpainter
    print("Device:", device)
    pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(os.getcwd(),"../models/sdxl-base-1.0"), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True)
    pipe.safety_checker = None

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    pipe.to(device)
    print("Initializing SDXL refiner...")
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        os.path.join(os.getcwd(),"../models/sdxl-refiner-1.0"),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    refiner.register_to_config(requires_aesthetics_score=False)
    refiner.config.requires_aesthetics_score = False
    refiner.safety_checker = None
    refiner.enable_attention_slicing()
    refiner.to(device)

    await init_inpainter()

async def generate_latent(prompt) -> None:
    global pipe, latent, prompt_embeds, negative_prompt_embeds
    if pipe is None:
        print("SDXL pipeline not initialized")
        return

    latent = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=10, negative_guidance_scale=10, output_type="latent", width=512, height=512).images[0]

    print("image:", latent, type(latent))


async def generate(prompt, iterations, guidance_scale, negative_prompt, prompt_2, negative_prompt_2) -> dict:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    global selected_image, selected_prompt, latent, prompt_embeds, negative_prompt_embeds
    selected_prompt = prompt
    # await generate_latent(prompt)
    global pipe
    if pipe is None:
        print("SDXL pipeline not initialized")
        return {}

    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=iterations, guidance_scale=guidance_scale, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, width=512, height=512).images[0]

    print(f"image: {image}")

    global buffer
    buffer = BytesIO()
    selected_image = image
    selected_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": image_base64}

def extract_base64_data(data_url: str) -> str:
    # Remove header if present
    if ',' in data_url:
        b64 = data_url.split(',', 1)[1]
    else:
        b64 = data_url
    b64 = b64.strip().replace('\n', '').replace(' ', '')
    # Fix base64 padding
    pad = len(b64) % 4
    if pad:
        b64 += '=' * (4 - pad)
    return b64

async def refine(prompt, image, strength, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, callback_on_step_end=None):
    if strength == 0:
        return {"image_data": image}

    global selected_image, selected_prompt, latent, pipe, refiner, device
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image = extract_base64_data(image)
    b64 = extract_base64_data(image)
    image_bytes = base64.b64decode(b64)

    image = Image.open(BytesIO(image_bytes))
    image = load_image(image)

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))
    image = ImageOps.pad(image, (1024, 1024), color=(0, 0, 0, 0))
    image.save(os.path.join(os.getcwd(), "app/test_images/padded_image.png"))
    # image = remove(image)
    image = image.convert("RGB")

    if refiner is None:
        print("SDXL refiner pipeline not initialized")
        return {}

    global buffer

    refiner.register_to_config(requires_aesthetics_score=False)
    refiner.config.requires_aesthetics_score = False
    print(f"refiner.config.requires_aesthetics_score: {refiner.config.requires_aesthetics_score}")

    image = refiner(prompt=prompt, image=image,
    strength=strength,
    original_size=image.size,
    guidance_scale = guidance_scale,
    crop_coords_top_left=(0, 0),
    target_size=(1024, 1024),
    negative_prompt=negative_prompt,
    prompt_2=prompt_2,
    negative_prompt_2=negative_prompt_2,
    output_type="pil",).images[0]


    buffer = BytesIO()
    image.save(buffer, format="PNG")

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image_data": image_base64}

async def inpaint(image, prompt="red lattice pattern", mask=None, strength=0.5, guidance_scale = 7.5, negative_prompt=None, prompt_2=None, negative_prompt_2=None, alpha = 255, gaussian_noise = 0.5, noise_offset=0.0, blur=2, strict=False, reverse_mask=False, callback_on_step_end=None):

    global inpainter, inpaint_image_count, buffer, inpaint_mask_count, inpaint_alpha_mask_count
    if inpainter is None:
        await init_inpainter()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    b64 = extract_base64_data(image)
    image_bytes = base64.b64decode(b64)

    image = Image.open(BytesIO(image_bytes))

    b64 = extract_base64_data(mask)
    image_bytes = base64.b64decode(b64)

    mask = Image.open(BytesIO(image_bytes)).convert("L")

    composited_image = ImageChops.composite(image, mask.convert("RGBA"), mask)

    if gaussian_noise > 0:
        noise_arr = np.random.standard_normal((image.height, image.width, 3)) * (gaussian_noise // 255 + noise_offset)
        noise = Image.fromarray(noise_arr, mode='RGB').convert("L")
        noise = noise.convert("RGBA")
        noise = noise.filter(ImageFilter.GaussianBlur(radius=blur))
        composited_image = ImageChops.composite(noise,composited_image, mask)
        composited_image.save(os.path.join(os.getcwd(),"app/test_images/noise-image-mask.png"))

        image_2 = ImageChops.composite(image,noise, mask)
        image_2.save(os.path.join(os.getcwd(),"app/test_images/image-noise-mask.png"))

        noise.putalpha(mask)
        image_3 = ImageChops.composite(noise,image, mask)
        image_3.save(os.path.join(os.getcwd(),"app/test_images/alpha-noise-image-mask.png"))
        image_4 = ImageChops.composite(image,noise, mask)
        image_4.save(os.path.join(os.getcwd(),"app/test_images/alpha-image-noise-mask.png"))

    width_scale = 1024 / composited_image.width
    height_scale = 1024 / composited_image.height
    scale = min(width_scale, height_scale)
    composited_image = composited_image.resize((int(math.floor(composited_image.width * scale)), int(math.floor(composited_image.height * scale))))
    composited_image = ImageOps.pad(composited_image, (1024, 1024), color=(0, 0,0,0))

    composited_image.save(os.path.join(os.getcwd(),"app/test_images/composited_image.png"))

    if reverse_mask:
        mask = ImageOps.invert(mask)

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))

    image = ImageOps.pad(image, (1024, 1024), color=(0, 0, 0, 0))

    width_scale = 1024 / mask.width
    height_scale = 1024 / mask.height
    scale = min(width_scale, height_scale)
    mask = mask.resize((int(math.floor(mask.width * scale)), int(math.floor(mask.height * scale))))
    mask = ImageOps.pad(mask, (1024, 1024), color=(0))

    image = composited_image if strict else image

    image = image.convert("RGB")

    print(f"Image dimensions: {image.width}x{image.height}")
    print(f"Mask dimensions: {mask.width}x{mask.height}")

    new_image = cast(StableDiffusionXLInpaintPipeline, inpainter)(prompt=prompt, image=image, mask_image=mask, strength=strength, guidance_scale=guidance_scale,
    negative_prompt=negative_prompt,
    crop_coords_top_left=(0, 0),
    prompt_2=prompt_2,
    negative_prompt_2=negative_prompt_2,
    output_type="pil",
    original_size=image.size,
    target_size=(1024, 1024),
    callback_on_step_end=callback_on_step_end).images[0]


    # zeros = np.zeros((new_image.height, new_image.width)).astype(np.uint8)
    # new_image_greyscale = new_image.convert("L")
    # new_image_greyscale.save(os.path.join(os.getcwd(),"app/test_images/new_image_greyscale.png"))
    # new_image_greyscale_arr = np.asarray(new_image_greyscale)
    # print(f"new_image_greyscale_arr: {new_image_greyscale_arr[700:749]}")
    # new_alpha_arr = np.heaviside(np.asarray(new_image.convert("L")), zeros).astype(np.uint8) * 255
    # new_alpha_mask = Image.fromarray(new_alpha_arr, mode="L")
    # new_alpha_mask.save(os.path.join(os.getcwd(),"app/test_images/new_alpha_mask.png"))
    # new_image = new_image.convert("RGBA")
    # new_image.putalpha(new_alpha_mask)

    # print(f"alpha_arr: {alpha_arr}")
    # alpha_mask = Image.fromarray(alpha_arr, mode="L")
    # alpha_mask.save(os.path.join(os.getcwd(),"app/test_images/alpha_mask.png"))
    # print(f"alpha_mask: {alpha_mask}")


    new_image.save(os.path.join(os.getcwd(),"app/test_images/new_image.png"))
    print(f"image.size: {new_image.size}")
    alpha_arr = np.ones((new_image.height, new_image.width)).astype(np.uint8) * alpha
    alpha_mask = Image.fromarray(alpha_arr, mode="L")
    alpha_mask.save(os.path.join(os.getcwd(),"app/test_images/alpha_mask.png"))
    new_image = new_image.convert("RGBA")
    new_image.putalpha(alpha_mask)
    new_image.save(os.path.join(os.getcwd(),"app/test_images/new_image_with_alpha.png"))
    new_image = ImageChops.composite(image.convert("RGBA"), new_image, ImageChops.invert(mask))

    # new_image = ImageChops.composite(image, new_image, new_alpha_mask)

    new_image.save(os.path.join(os.getcwd(),f"app/test_images/inpaint-{inpaint_image_count}.png"))
    inpaint_image_count += 1

    buffer = BytesIO()
    new_image.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image_data": b64}


def train_one():
    pass
