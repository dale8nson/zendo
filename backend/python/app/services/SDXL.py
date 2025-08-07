from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline, AutoPipelineForImage2Image, StableDiffusionUpscalePipeline, DDPMScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel, DiffusionPipeline

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
import cv2


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
controlnet = None

class GenerateRequest(BaseModel):
    prompt: str
    iterations: int
    guidance_scale: float = 7.5
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str
    ipAdapterImage: str | None

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

        pipe = pipe.to(device)

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

    if torch.cuda.is_available():
        refiner.enable_model_cpu_offload()

    refiner.config.output_type = "pil"
    refiner.safety_checker = None
    refiner.enable_attention_slicing()
    refiner = refiner.to(device)

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

    if torch.cuda.is_available():
        inpainter.enable_model_cpu_offload()

    inpainter = inpainter.to(device)


async def init_controlnet():
    global pipe, controlnet

    if pipe is None:
        await init_SDXL()

    controlnet_model = ControlNetModel.from_pretrained(
        os.path.join(os.getcwd(), "../models/controlnet-canny-sdxl-1.0"),
        torch_dtype=torch.float32
    )
    vae = pipe.vae
    controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
        os.path.join(os.getcwd(), "../models/sdxl-base-1.0"),
        controlnet=controlnet_model,
        vae=vae,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    controlnet.safety_checker = None

    if torch.cuda.is_available():
        controlnet.enable_model_cpu_offload()

    controlnet = controlnet.to(device)

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

async def generate_latent(image: Image.Image, pipeline: DiffusionPipeline, timestep: torch.Tensor) -> torch.Tensor:
    global pipe, latent, prompt_embeds, negative_prompt_embeds
    if pipe is None:
        print("SDXL pipeline not initialized")
        return

    latent = torch.tensor(np.asarray(image)).unsqueeze(0).to(device, dtype=torch.float32)
    print(f"latent.size(): {latent.size()}")
    latent = latent.movedim(3, 1)
    latent = pipeline.vae.encode(latent).latent_dist.sample()

    noise = torch.randn_like(latent)
    latent = pipeline.scheduler.add_noise(latent, noise, timestep)

    return latent


async def generate(prompt, iterations, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, ipAdapterImage=None) -> dict:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    global selected_image, selected_prompt, latent, prompt_embeds, controlnet, negative_prompt_embeds, inpainter, refiner
    selected_prompt = prompt
    # await generate_latent(prompt)
    global pipe
    if pipe is None:
        await init_SDXL()

    if ipAdapterImage is not None:

        if controlnet is None:
            await init_controlnet()

    if inpainter is not None:
        del inpainter

    if refiner is not None:
        del refiner

        # controlnet = cast(controlnet, StableDiffusionXLControlNetPipeline)
    if ipAdapterImage is not None:
        b64 = extract_base64_data(ipAdapterImage)
        image_bytes = base64.b64decode(b64)

        ipAdapterImage = Image.open(BytesIO(image_bytes)).convert("RGB")

        print(f"ipAdapterImage: {ipAdapterImage}")
        width_scale = 1024 / ipAdapterImage.width
        height_scale = 1024 / ipAdapterImage.height
        scale = min(width_scale, height_scale)

        ipAdapterImage = ipAdapterImage.resize((int(ipAdapterImage.width * scale), int(ipAdapterImage.height * scale)))

        # ipAdapterImage = ImageOps.pad(ipAdapterImage, (1024, 1024), color=(0,0,0)).convert("RGB")

        print(f"ipAdapterImage: {ipAdapterImage}")
        ipAdapterImage = np.asarray(ipAdapterImage)
        ipAdapterImage = cv2.Canny(ipAdapterImage, 100, 200)
        ipAdapterImage = ipAdapterImage[:, :, None]
        ipAdapterImage = np.concatenate([ipAdapterImage, ipAdapterImage, ipAdapterImage], axis=2)
        ipAdapterImage = Image.fromarray(ipAdapterImage)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = controlnet(prompt=prompt, negative_prompt=negative_prompt, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, image=ipAdapterImage, guidance_scale=guidance_scale, num_inference_steps=iterations, controlnet_conditioning_scale=0.65).images[0]

        image.save(os.path.join(os.getcwd(), f"app/test_images/controlnet-{datetime.utcnow()}.png"))

        buf = BytesIO()
        selected_image = image
        selected_image.save(buf, format="PNG")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"image": image_base64}

    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=iterations, guidance_scale=guidance_scale, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, ip_adapter_image=ipAdapterImage, width=1024, height=1024).images[0]

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

    global selected_image, selected_prompt, latent, pipe, refiner, device, test_filepath, inpainter

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if inpainter is not None:
        del inpainter

    b64 = extract_base64_data(image)
    image_bytes = base64.b64decode(b64)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_original.png"))

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))

    if refiner is None:
        await init_refiner()

    refiner = cast(StableDiffusionXLImg2ImgPipeline, refiner)

    image = image.convert("RGB")
    image = ImageOps.pad(image, (1024, 1024), color=(0, 0, 0))

    denoising_end = 1.0 - strength
    refiner.scheduler.set_timesteps(inference_steps)
    t_index = int(denoising_end * inference_steps)
    if t_index >= len(refiner.scheduler.timesteps):
        t_index = len(refiner.scheduler.timesteps) - 1

    timestep = refiner.scheduler.timesteps[t_index]
    print(f"timestep: {timestep}")
    latent = torch.tensor(np.asarray(image)).unsqueeze(0).to(device, dtype=torch.float32)
    print(f"latent.size(): {latent.size()}")
    latent = latent.movedim(3, 1)

    latent = refiner.vae.encode(latent).latent_dist.sample()

    latents_mean = latents_std = None

    if hasattr(refiner.vae.config, "latents_mean") and refiner.vae.config.latents_mean is not None:
        latents_mean = torch.tensor(refiner.vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(refiner.vae.config, "latents_std") and refiner.vae.config.latents_std is not None:
        latents_std = torch.tensor(refiner.vae.config.latents_std).view(1, 4, 1, 1)

    if torch.cuda.is_available or torch.backends.mps.is_available()():
        if hasattr(refiner, "final_offload_hook") and refiner.final_offload_hook is not None:
            refiner.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
    else:
        # make sure the VAE is in float32 mode, as it overflows in float16
        if refiner.vae.config.force_upcast:
            latent = latent.float()
            refiner.vae.to(dtype=torch.float32)

    if latents_mean is not None and latents_std is not None:
        latents_mean = latents_mean.to(device=device, dtype=dtype)
        latents_std = latents_std.to(device=device, dtype=dtype)
        latent = (latent - latents_mean) * refiner.vae.config.scaling_factor / latents_std
    else:
        latent = refiner.vae.config.scaling_factor * latent

    print(f"latent.size(): {latent.size()}")

    noise = torch.randn_like(latent)

    latent = refiner.scheduler.add_noise(latent, noise, torch.tensor([timestep]))

    print(f"type(latent): {type(latent)}")

    print(f"image: {image}")

    denoising_start = 1.0 - strength

    output = cast(StableDiffusionXLImg2ImgPipeline, refiner)(
        prompt=prompt,
        image=image,
        latents=latent,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        crop_coords_top_left=(0, 0),
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        output_type="pil",
        original_size=image.size,
        target_size=(1024, 1024),
        denoising_start=denoising_start,
        denoising_end=1.0,
        num_images_per_prompt = 1,
        num_inference_steps = inference_steps,
        callback_on_step_end=callback_on_step_end)

    print(type(output))

    if hasattr(output, "images"):
        output = output.images[0]
    else:
        raise ValueError("Refiner output does not include images; possibly returned latents.")

    output = ImageOps.pad(output, (1024, 1024), color=(128, 128, 128)).convert("RGB")

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

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"image_data": image_base64}


async def inpaint(image, prompt="", mask_data = [], strength=0.5, inference_steps=50, guidance_scale = 7.5, use_refiner=False, inpaint_refiner_ratio=0.85, inpaint_refiner_inference_steps=5,  inpaint_refiner_guidance_scale=7.5, negative_prompt=None, prompt_2=None, negative_prompt_2=None, refiner_prompt=None, refiner_negative_prompt=None, refiner_prompt_2=None, refiner_negative_prompt_2=None, callback_on_step_end=None):

    global inpainter, inpaint_image_count, buffer, inpaint_mask_count, inpaint_alpha_mask_count, image_count, refiner, pipe
    if inpainter is None:
        await init_inpainter()

    if refiner is None:
        await init_refiner()

    if inpainter is not None:
        del inpainter

    if pipe is not None:
        del pipe

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

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"image_data": b64}
