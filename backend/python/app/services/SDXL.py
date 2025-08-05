from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline, AutoPipelineForImage2Image, StableDiffusionUpscalePipeline

import torch
from torchvision import transforms
import torchvision.transforms as T
from pydantic.main import BaseModel
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageEnhance
import os
import base64
from io import BytesIO
from typing import cast, List
import numpy as np
from diffusers.utils import load_image
# from concurrent.futures import ProcessPoolExecutor
from rembg import remove
import math
from datetime import datetime, timezone
import asyncio


cwd = os.getcwd()
print(f"cwd:{cwd}")
test_filepath = "app/test_images" if cwd.endswith("python") else "../test_images"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

pipe = None
refiner: StableDiffusionXLImg2ImgPipeline | None = None
inpainter: StableDiffusionXLInpaintPipeline | None = None
upscaler = None
selected_image = Image.Image | None
selected_prompt: str | None = None
negative_prompt: str = ""
latent = None
prompt_embeds = None
negative_prompt_embeds = None
buffer = BytesIO()
inpaint_image_count = 0
inpaint_mask_count = 0
inpaint_alpha_mask_count = 0
image_count = 0

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
    negative_prompt_2: str = ""


class InpaintRequest(BaseModel):
    image: str
    prompt: str
    masks: List[str]
    strength: float
    guidance_scale: float = 7.5,
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str = ""
    alpha: int = 255
    noise: float = 0.5
    noise_offset: float = 0.0
    blur: int = 2
    strict: bool = False
    reverse_mask: bool = False

async def to_async(model, dev) -> None:
    model.to(dev)

async def init_SDXL():
    print("Initializing SDXL pipeline...")
    global pipe, device, refiner, inpainter
    print("Device:", device)
    if pipe is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(os.getcwd(),"../models/sdxl-base-1.0"), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True)
        pipe.safety_checker = None

        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        pipe.to(device)

async def init_refiner(model_path = None):
    global refiner
    print("Initializing SDXL refiner...")
    if model_path is None:
        model_path = os.path.join(os.getcwd(),"../models/sdxl-refiner-1.0")
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )

    refiner.config.output_type = "pil"
    refiner.safety_checker = None
    refiner.enable_attention_slicing()
    refiner.to(device)

    print(refiner.scheduler.config)
    print(refiner.config)

async def init_inpainter(model_path=None):
    if model_path is None:
        model_path = "../models/stable-diffusion-xl-inpainting-1.0"
    global refiner, inpainter
    print("Initializing SDXL inpainter...")
    inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
        os.path.join(os.getcwd(),model_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant = "fp16" if torch.cuda.is_available() else None,
        use_safetensors=True
    )

    inpainter.register_to_config(requires_aesthetics_score=False)
    inpainter.safety_checker = None
    inpainter.enable_attention_slicing()
    inpainter.enable_vae_slicing()
    inpainter.vae.enable_tiling(False)
    inpainter.to(device)


async def get_pipe():
    global pipe
    if pipe is None:
        await init_SDXL()
    return pipe

async def get_refiner():
    global refiner
    if pipe is None:
        await init_refiner()
    return refiner

async def get_inpainter():
    global inpainter
    if inpainter is None:
        await init_inpainter()
    return inpainter

async def init_upscaler():
    global upscaler
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(os.path.join(os.getcwd(), "../models/stable-diffusion-x4-upscaler"), torch_dtype=torch.float16 if torch.cuda.is_available else torch.float32, variant = "fp16" if torch.cuda.is_available() else None,
    use_safetensors=True)
    upscaler.safety_checker = None
    upscaler.enable_attention_slicing()
    upscaler.enable_vae_slicing()
    upscaler.enable_vae_tiling()
    upscaler.to(device)

async def generate_latent(prompt, negative_prompt, prompt_2, negative_prompt_2, inference_steps=50, guidance_scale=7.5) -> None:
    global pipe, latent, prompt_embeds, negative_prompt_embeds
    if pipe is None:
        print("SDXL pipeline not initialized")
        return

    latent = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=10, negative_guidance_scale=10, output_type="latent", width=1024, height=1024).images[0]

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

    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=iterations, guidance_scale=guidance_scale, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, width=1024, height=1024).images[0]

    print(f"image: {image}")
    image.save(os.path.join(os.getcwd(), f"app/test_images/generated-image-{datetime.now(timezone.utc)}.png"))
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

k = 0

async def refine(prompt, image, strength, inference_steps, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, callback_on_step_end=None):

    global selected_image, selected_prompt, latent, pipe, refiner, device, test_filepath
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    b64 = extract_base64_data(image)
    image_bytes = base64.b64decode(b64)

    # original_image = Image.open(BytesIO(image_bytes))
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    # image = load_image(image)



    image.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_original.png"))
    # original_size = image.size
    # image = image.resize((1024, 1024))
    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))
    # image = ImageOps.cover(image, (1024, 1024))

    # image.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_step_{step}_covered.png"))
    # step += 1
    image = ImageOps.pad(image, (1024, 1024), color=(128, 128, 128)).convert("RGB")

    # image.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_step_{step}_padded.png"))

    if refiner is None:
        await init_refiner()

    global buffer

    refiner = cast(StableDiffusionXLImg2ImgPipeline, refiner)

    image = image.convert("RGB")
    image = ImageOps.pad(image, (1024, 1024), color=(0, 0, 0))
    # image = np.array(image).astype(np.float32) / 255.0
    # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # image = 2.0 * image - 1.0
    # image = image.to(device)

    # vae = refiner.vae
    # scheduler = refiner.scheduler
    # # transform = T.Compose([T.Resize((1024, 1024)), T.ToTensor(), T.Lambda(lambda x: x * 2.0 - 1.0)])

    # # image_tensor = transform(image).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     latent = vae.encode(image).latent_dist.sample()
    #     latent = latent * vae.config.scaling_factor

    #     # latents = [ m.mean * 0.18215 for m in vae.encode(image_tensor)]


    # noise = torch.randn_like(latent)
    # timestep = torch.tensor([scheduler.timesteps[len(scheduler.timesteps) // 2]], device=latent.device)
    # noisy_latent = scheduler.add_noise(latent, noise, timestep)

    # print("Latent shape:", latent.shape)
    # assert latent.numel() > 0, "Latent tensor is empty!"
    #

    print(f"image: {image}")

    # latent = refiner(
    #     prompt=prompt,
    #     image=image,
    #     strength=strength,
    #     guidance_scale=0.1,
    #     negative_prompt=negative_prompt,
    #     crop_coords_top_left=(0, 0),
    #     prompt_2=prompt_2,
    #     negative_prompt_2=negative_prompt_2,
    #     output_type="latent",
    #     original_size=image.size,
    #     target_size=(1024, 1024),
    #     denoise_start=0.0,
    #     denoise_end=0.6,
    #     num_inference_steps = inference_steps,
    #     # num_inference_steps = 30,
    #     callback_on_step_end=callback_on_step_end).images[0]

    # print(f"output.size: {latent.size()}, type(output): {type(latent)}")

    # batch_size = 2  # number of prompts

    # # If guidance is used, double everything

    # if refiner.do_classifier_free_guidance:
    #     num_images_per_prompt = 1  # default
    #     batch_size *= 2

    # # Now repeat the latents to match expected batch size
    # if latent.shape[0] != batch_size:
    #     latent = latent.repeat_interleave(batch_size // latent.shape[0], dim=0)
    #
    # latent = latent.unsqueeze(0)

    output = cast(StableDiffusionXLImg2ImgPipeline, refiner)(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        crop_coords_top_left=(0, 0),
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        output_type="pil",
        original_size=image.size,
        target_size=(1024, 1024),
        denoise_start=0.6,
        denoise_end=1.0,
        num_images_per_prompt = 1,
        num_inference_steps = inference_steps,
        callback_on_step_end=callback_on_step_end)

    print(type(output))

    if hasattr(output, "images"):
        output = output.images[0]
    else:
        raise ValueError("Refiner output does not include images; possibly returned latents.")

    output.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img-{datetime.utcnow()}.png"))

    print(f"Final output type: {type(output)}")
    print("output keys:", output.__dict__.keys() if hasattr(output, "__dict__") else dir(output))
    print("Output.images type:", type(output.images[0]) if hasattr(output, "images") else "No images")

    # output = output.resize(original_size)
    # output.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_final_output_resized.png"))

    buffer = BytesIO()
    output.save(buffer, format="PNG")

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image_data": image_base64}


async def inpaint(image, prompt="", mask_data = [], strength=0.5, inference_steps=50, guidance_scale = 7.5, use_refiner=False, inpaint_refiner_ratio=0.85, inpaint_refiner_inference_steps=5,  inpaint_refiner_guidance_scale=7.5, negative_prompt=None, prompt_2=None, negative_prompt_2=None, refiner_prompt=None, refiner_negative_prompt=None, refiner_prompt_2=None, refiner_negative_prompt_2=None, callback_on_step_end=None):

    global inpainter, inpaint_image_count, buffer, inpaint_mask_count, inpaint_alpha_mask_count, image_count, refiner
    if inpainter is None:
        await init_inpainter()

    if refiner is None:
        await init_refiner()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    b64 = extract_base64_data(image)
    image_bytes = base64.b64decode(b64)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    composite_mask = Image.new("L", image.size, 0)

    mask_count = 0

    mask_data = sorted(mask_data, key=lambda x: x["area"], reverse=True)

    for data in mask_data:
        mask = data["mask"] if data["include"] == True else data["inverted_mask"] if data["exclude"] == True else None
        if mask is None:
            continue
        step = 1
        b64 = extract_base64_data(mask)
        image_bytes = base64.b64decode(b64)

        mask = Image.open(BytesIO(image_bytes))
        mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-original.png"))
        step += 1

        alpha_channel = mask.getchannel("A").convert("L")
        # alpha_channel.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step-{step}-alpha_channel.png"))
        step += 1
        # print(f"alpha_channel: {alpha_channel}")

        rgb = mask.convert("RGB")
        rgb.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-rgb.png"))

        # width_scale = 1024 / mask.width
        # height_scale = 1024 / mask.height
        # scale = min(width_scale, height_scale)

        binary_mask = alpha_channel.point(lambda p: 255 if p > 0 else 0).convert("L")
        # print(f"binary_mask: {binary_mask}")
        binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-binary.png"))
        step += 1
        # binary_mask = binary_mask.resize((int(math.floor(mask.width * scale)), int(math.floor(mask.height * scale))))
        # binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step-{step}-resized.png"))
        step += 1
        # binary_mask = ImageOps.pad(binary_mask, (1024, 1024), color=(0))
        print(f"binary_mask: {binary_mask}")
        # binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step--{step}-padded.png"))
        step += 1

        inverted_binary = ImageOps.invert(binary_mask)
        inverted_binary.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-inverted-binary.png"))

        composite_mask.paste(binary_mask if data["include"] else ImageOps.invert(binary_mask), (data["bbox"][0], data["bbox"][1]), binary_mask)

        # composite_mask = ImageChops.lighter(composite_mask, binary_mask)
        # composite_mask = ImageChops.logical_or(composite_mask, binary_mask)

        composite_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-composite.png"))
        print(f"composite_mask: {composite_mask}")

        mask_count += 1


    # composite_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-composite_binary.png"))

    composite_mask = composite_mask.convert("L")
    print(f"composite_mask: {composite_mask}")
    # composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/mask-{mask_count}-composite_L.png"))
    lut = [255 if i > 0 else 0 for i in range(256)]
    composite_mask = composite_mask.point(lut).convert('L')
    composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/mask-composite-binary.png"))

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))
    composite_mask = composite_mask.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))

    image = ImageOps.pad(image, (1024, 1024), color=0)
    composite_mask = ImageOps.pad(composite_mask, (1024, 1024), color=0)

    image = image.convert("RGB")
    assert image.size == composite_mask.size
    print(f"image: {image}")
    # image.save(os.path.join(os.getcwd(),f"{test_filepath}/image-{image_count}-converted.png"))
    print(f"Image dimensions: {image.width}x{image.height}")
    # print(f"Mask dimensions: {mask.width}x{mask.height}")
    #
    image = image.copy()
    enhancer = ImageEnhance.Contrast(image)
    enhancer.enhance(0.5)

    composite_mask = composite_mask.copy()
    composite_mask = composite_mask.filter(ImageFilter.GaussianBlur(3))

    composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-mask-final-input.png"))
    image.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-image-final-input.png"))

    assert image.mode == "RGB", f"Unexpected image mode: {image.mode}"
    assert composite_mask.mode == "L", f"Unexpected mask mode: {composite_mask.mode}"
    assert image.size == composite_mask.size, f"Size mismatch: {image.size} vs {composite_mask.size}"

    output = cast(StableDiffusionXLInpaintPipeline, inpainter)(
        prompt=prompt,
        image=image,
        mask_image=composite_mask,
        num_inference_steps=inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        crop_coords_top_left=(0, 0),
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        original_size=image.size,
        target_size=image.size,
        output_type="latent" if use_refiner else "pil",
        denoising_end=inpaint_refiner_ratio if use_refiner else 1.0,
        callback_on_step_end=callback_on_step_end).images[0]

    print(f"type(output): {type(output)}")

    if use_refiner:
        output = output.unsqueeze(0)
        output = cast(StableDiffusionXLImg2ImgPipeline, refiner)(
            prompt=prompt,
            image=image,
            latents=output,
            mask_image=composite_mask,
            guidance_scale=inpaint_refiner_guidance_scale,
            negative_prompt=refiner_negative_prompt,
            crop_coords_top_left=(0, 0),
            prompt_2=refiner_prompt_2,
            negative_prompt_2=refiner_negative_prompt_2,
            original_size=image.size,
            target_size=image.size,
            output_type="pil",
            denoising_start=inpaint_refiner_ratio,
            num_inference_steps=inpaint_refiner_inference_steps,
            callback_on_step_end=callback_on_step_end).images[0]

    output = output.convert("RGBA")
    # output.putalpha(alpha_mask)

    output.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-image-final-output.png"))
    print(f"image.size: {output.size}")
    # alpha_arr = np.ones((output.height, output.width)).astype(np.uint8) * alpha
    # alpha_mask = Image.fromarray(alpha_arr, mode="L")
    # alpha_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/alpha_mask.png"))
    # output = output.convert("RGBA")
    # output.putalpha(alpha_mask)
    # new_image.save(os.path.join(os.getcwd(),f"{test_filepath}/new_image_with_alpha.png"))

    output.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-{datetime.utcnow()}.png"))

    inpaint_image_count += 1

    buffer = BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('utf-8')

    image_count += 1
    mask_count += 1

    return {"image_data": b64}



async def upscale(image_data, prompt):
    global upscaler

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if upscaler is None:
        await init_upscaler()

    b64 = extract_base64_data(image_data)
    image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    size = max(image.width, image.height)
    image= ImageOps.pad(image,(size, size), color=(0,0,0))
    image = image.resize((512, 512))
    upscaled_image = upscaler(prompt=prompt, image=image).images[0]

    upscaled_image.save(os.path.join(os.getcwd(), f"app/test_images/upscale-{datetime.now(timezone.utc)}.png"))

    buf = BytesIO()
    upscaled_image.save(buf, format="PNG")
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode("utf8")

    return {"image_data": image_data}


# def train_one(image_data: str, prompt: str):
#     bytes = base64.b64decode(image_data)
#     image = Image.open(BytesIO(bytes))

async def test_refiner():
    global refiner
    if refiner is None:
        await init_refiner(model_path=os.path.join(os.getcwd(), "../../../models/sdxl-refiner-1.0"))
    prompt = "young woman bending over posing in her white cotton panties"
    image = Image.open(os.path.join(os.getcwd(), "../test_images/img2img_step_7_output.png"))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('utf-8')
    strength = 0.2
    guidance_scale = 7.5
    prompt_2 = "crisp, 8k, high definition, high resolution, high quality, full-frame camera, smooth, sharp focus, long depth of field,"
    negative_prompt = "cartoon, art, low quality, pixelated, low resolution, distorted, deformities, fuzzy, blurry, suspenders, garter belt, bokeh, collar, headdress, baseball cap"
    negative_prompt_2 = "cartoon, art, low quality, pixelated, low resolution, distorted, deformities, fuzzy, blurry, suspenders, garter belt, bokeh, collar, headdress, baseball cap"

    data = await refine(prompt, image, strength, guidance_scale, negative_prompt, prompt_2, negative_prompt_2)
    image_bytes = base64.b64decode(data["image_data"])
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    image.save(os.path.join(os.getcwd(), "../test_images/test_result.png"))


async def test_inpainter():
    global inpainter
    if inpainter is None:
        await init_inpainter(model_path=os.path.join(os.getcwd(), "../../../models/stable-diffusion-xl-inpainting-1.0"))
    prompt = "busty nude young woman posing in white cotton fullback panties in front of a couch. Her fullback panties are thin and tight and have zigzag stitch and picot elastic"
    image = Image.open(os.path.join(os.getcwd(), "../test_images/inpaint-d.png")).convert("RGBA")
    print(f"image {image}")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('utf-8')
    image = extract_base64_data(image)
    print(f"base64 image: {image[-39:-1]}")
    mask = Image.open(os.path.join(os.getcwd(), "../test_images/mask-composite_binary.png")).convert("RGBA")
    print(f"mask: {mask}")
    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    buffer.seek(0)
    mask = base64.b64encode(buffer.read()).decode('utf-8')
    mask = extract_base64_data(mask)
    print(f"base64 mask: {mask[-39:-1]}")
    strength = 0.2
    guidance_scale = 7.5
    prompt_2 = "crisp, 8k, high definition, high resolution, high quality, full-frame camera, smooth, sharp focus, long depth of field,"
    negative_prompt = "cartoon, art, low quality, pixelated, low resolution, distorted, deformities, fuzzy, blurry, suspenders, garter belt, pantyhose"
    negative_prompt_2 = "cartoon, art, low quality, pixelated, low resolution, distorted, deformities, fuzzy, blurry, suspenders, garter belt, pantyhose"

    data = await inpaint(prompt, image, [mask], strength, guidance_scale, negative_prompt, prompt_2, negative_prompt_2)
    image_bytes = base64.b64decode(data["image_data"])
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    print(f"imag: {image}")
    image.save(os.path.join(os.getcwd(), "../test_images/inpaint_test_result.png"))

if __name__ == "__main__":
    # asyncio.run(test_refiner())
    asyncio.run(test_inpainter())
