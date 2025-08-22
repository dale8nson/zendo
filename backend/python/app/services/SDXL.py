from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,  StableDiffusionXLControlNetPipeline, ControlNetModel, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

from transformers import CLIPTokenizer

import torch
import torchvision.transforms as T
from safetensors.torch import load_file

from pydantic.main import BaseModel
from PIL import Image, ImageOps, ImageEnhance
import os
import base64
from io import BytesIO
from typing import cast, List, Any, TypedDict
import numpy as np
from rembg import remove
from datetime import datetime, timezone
import cv2
import re


class LayerImage(TypedDict):
    bbox: List[float]
    imageData: str

class Layer(BaseModel):
    selected: bool
    label: str
    visible: bool
    opacity: float
    currentLayerHistoryIndex: int
    history: List[Any]


cwd = os.getcwd()
print(f"cwd:{cwd}")
test_filepath = "app/test_images" if cwd.endswith("python") else "../test_images"

def is_available():
    return False

torch.cuda.is_available = is_available

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
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    iterations: int
    guidance_scale: float
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str
    ip_adapter_image: str | List[Any] | None
    use_face_id: bool
    bbox: List[float] | None
    remove_background: bool

class RefineRequest(BaseModel):
    prompt: str
    image: str
    strength: float
    guidance_scale: float
    negative_prompt: str = negative_prompt
    prompt_2: str
    negative_prompt_2: str = ""
    remove_background: bool


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
    remove_background: bool


class TokenizeRequest(BaseModel):
    text: str


def is_available():
    return False


torch.cuda.is_available = is_available

async def init_SDXL():
    print("Initializing SDXL pipeline...")
    global pipe, device, refiner, inpainter
    print("Device:", device)
    if pipe is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(os.getcwd(),"../models/sdxl-base-1.0"), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
           # device_map="balanced",
           use_safetensors=True)
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
        # device_map="balanced",
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

    controlnet = StableDiffusionXLControlNetPipeline.from_pipe(
        pipe,
        controlnet=controlnet_model,
        torch_dtype=torch.float32,
    )
    controlnet.safety_checker = None

    if torch.cuda.is_available():
        controlnet.enable_model_cpu_offload()

    controlnet.to(device)


async def init_tokenizer():
    global tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "../models/sdxl-base-1.0"), subfolder="tokenizer")


async def load_embeddings():
    global pipe, refiner, inpainter, controlnet
    print("Loading embeddings")
    path = os.path.join(os.getcwd(), "../models/user")

    dir_list = [d for d in filter(lambda f: os.path.isdir(os.path.join(path,f)), os.listdir(path))]

    print(f"dir_list: {dir_list}")

    for sd in dir_list:
        dir_path = os.path.join(path, sd)
        print(f"Searching {dir_path}...")
        files = [f for f in filter(lambda s: re.match(f"{sd}(_2)?\\.safetensors", s), os.listdir(dir_path))]


        if len(files) < 2:
            files = [f for f in filter(lambda n: re.match(f"{dir}(_2)?-steps-(\\d+).safetensors", n), dir_list)]

            if len(files) < 2:
                continue
            else:
                files = sorted(files, key=lambda f: int(f.split('-')[2].split('.')[0]))[-2:-1]

        print(f"files: {files}")


        for file in files:

            print(f"loading {file}...", end=' ')
            try:
                if isinstance(pipe, StableDiffusionXLPipeline):
                    pipe.unload_textual_inversion()
                    pipe.load_textual_inversion(file)
            except:
                pass

            try:
                if isinstance(refiner, StableDiffusionXLImg2ImgPipeline):
                    refiner.unload_textual_inversion()
                    refiner.load_textual_inversion(file)
            except:
                pass

            try:
                if isinstance(inpainter, StableDiffusionXLInpaintPipeline):
                    inpainter.unload_textual_inversion()
                    inpainter.load_textual_inversion(file)
            except:
                pass
            try:
                if isinstance(controlnet, StableDiffusionXLControlNetPipeline):
                    controlnet.unload_textual_inversion()
                    controlnet.load_textual_inversion(file)
            except:
                pass

            print("done.")

async def get_pipe():
    global pipe
    return pipe

async def get_refiner():
    global refiner
    return refiner

async def get_inpainter():
    global inpainter
    return inpainter

async def get_controlnet():
    global controlnet
    return controlnet

async def generate_latent(image: Image.Image, pipeline: DiffusionPipeline, timestep: torch.Tensor) -> torch.Tensor:
    global pipe, latent, prompt_embeds, negative_prompt_embeds
    if pipe is None:
        await init_SDXL()

    latent = torch.tensor(np.asarray(image)).unsqueeze(0).to(device, dtype=torch.float32)
    print(f"latent.size(): {latent.size()}")
    latent = latent.movedim(3, 1)
    latent = pipeline.vae.encode(latent).latent_dist.sample()

    noise = torch.randn_like(latent)
    latent = pipeline.scheduler.add_noise(latent, noise, timestep)

    return latent

async def generate(prompt, iterations, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, ip_adapter_image=None, use_face_id=False, bbox=None, remove_background=False) -> dict:

    global selected_image, pipe, selected_prompt, latent, prompt_embeds, controlnet, negative_prompt_embeds, inpainter, refiner

    selected_prompt = prompt

    if inpainter is not None:
        inpainter.to("cpu")
        inpainter = None

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if refiner is None:
        await init_refiner()
        refiner = cast(StableDiffusionXLImg2ImgPipeline, refiner)

    if pipe is None:
        await init_SDXL()

    if ip_adapter_image is not None:

        if controlnet is None:
            await init_controlnet()
            controlnet = cast(StableDiffusionXLControlNetPipeline, controlnet)

    await load_embeddings()

    if ip_adapter_image is not None:
        if use_face_id and pipe is not None:
            pipe.load_ip_adapter(
                '../models/IP-Adapter-FaceID',
                subfolder=".",
                weight_name="pytorch_model.bin",
                use_safetensors=True,
                slice_size=1
            )

    if ip_adapter_image is not None:
        print(f"type(ip_adapter_image): {type(ip_adapter_image)}")
        if type(ip_adapter_image) == list:
            layers = ip_adapter_image

            image = Image.new("RGBA", (1024, 1024))
            root_layer = [l if l["label"] == "root" else {} for l in layers][0]
            root_layer_history_index = root_layer["currentLayerHistoryIndex"]
            root_bbox = root_layer["history"][root_layer_history_index]["bbox"]
            rx1, ry1, rx2, ry2 = root_bbox
            root_layer_width = rx2 - rx1
            root_layer_height = ry2 - ry1
            scale_x = 1024 / root_layer_width
            scale_y = 1024 / root_layer_height
            scale = min(scale_x, scale_y)

            for layer in layers:

                history_index = layer["currentLayerHistoryIndex"]
                history = layer["history"]
                image_data = history[history_index]["imageData"]
                print(f"image_data: {image_data[0:100]}")
                bbox = history[history_index]["bbox"]

                bbox = [int(n * scale) for n in bbox]
                print(f"bbox: {bbox}, bbox width: {bbox[2] - bbox[0]} bbox height: {bbox[3] - bbox[1]}")

                b64 = extract_base64_data(image_data)
                image_bytes = base64.b64decode(b64)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                layer_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
                layer_image = layer_image.resize((w, h))

                print(f"layer_image width: {layer_image.width} layer_image height: {layer_image.height}")

                image.paste(layer_image, box=bbox)

            ip_adapter_image = image

        else:
            b64 = extract_base64_data(ip_adapter_image)
            image_bytes = base64.b64decode(b64)

            ip_adapter_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if remove_background:

            latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(device)
            latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(device)

            num_latent_channels:int = pipe.unet.config.in_channels
            vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)

            foreground = remove(ip_adapter_image, bgcolor=(0, 0, 0, 0))

            alpha = foreground.getchannel('A').convert('RGB')
            alpha_tensor = T.PILToTensor(alpha)

            foreground = foreground.convert('RGB')
            image = pipe.image_processor.preprocess(foreground)
            image.to(device)
            image = image.float()
            fg_latent = pipe.vae.encode(image).latent_dist.sample()
            bg_latent = torch.randn(shape=fg_latent.shape)
            alpha_latent = pipe.vae.encode(alpha_tensor).latent_dist.sample()
            bg_latent.to(device)
            foreground.to(device)
            alpha_latent.to(device)

            ip_adapter_image = torch.where(alpha_latent == 0, bg_latent, fg_latent)

            print(f'image.shape: {image.shape}')

            pipe.vae.to(dtype=torch.float32)


        print(f"ipAdapterImage: {ip_adapter_image}")
        width_scale = 1024 / ip_adapter_image.width
        height_scale = 1024 / ip_adapter_image.height
        scale = min(width_scale, height_scale)

        ip_adapter_image = ip_adapter_image.resize((int(ip_adapter_image.width * scale), int(ip_adapter_image.height * scale)))

        print(f"ipAdapterImage: {ip_adapter_image}")
        ip_adapter_image = np.asarray(ip_adapter_image)
        ip_adapter_image = cv2.Canny(ip_adapter_image, 100, 200)
        ip_adapter_image = ip_adapter_image[:, :, None]
        ip_adapter_image = np.concatenate([ip_adapter_image, ip_adapter_image, ip_adapter_image], axis=2)
        ip_adapter_image = Image.fromarray(ip_adapter_image)

        ip_adapter_image = ImageOps.pad(ip_adapter_image, (1024, 1024), color=(0, 0, 0))

        # latent = pipe.vae.encode(ip_adapter_image).latent_dist.sample()

        # ip_adapter_image = torch.where(alpha_tensor == 0, bg_latent, ip_adapter_image)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = controlnet(prompt=prompt, negative_prompt=negative_prompt, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, image=ip_adapter_image, guidance_scale=guidance_scale, num_inference_steps=iterations,
        output_type='latent',
        denoising_end=0.8,
        controlnet_conditioning_scale=0.65).images[0]

        # image.save(os.path.join(os.getcwd(), f"app/test_images/controlnet-{datetime.utcnow()}.png"))

        # buf = BytesIO()
        # selected_image = image
        # selected_image.save(buf, format="PNG")
        # buf.seek(0)
        # image_data = base64.b64encode(buf.read()).decode('utf-8')

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=iterations, guidance_scale=guidance_scale, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, ip_adapter_image=ip_adapter_image, width=1024, height=1024, denoising_end=0.8, output_type='latent').images[0]

    image = refiner( prompt=prompt, num_inference_steps=iterations, denoising_start=0.8, image=image).images[0]

    if use_face_id and pipe is not None:
        pipe.unload_ip_adapter()

    print(f"image: {image}")
    image.save(os.path.join(os.getcwd(), f"app/test_images/generated-image-{datetime.now(timezone.utc)}.png"))

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

async def refine(prompt, layers, strength, inference_steps, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, remove_background=False, callback_on_step_end=None):

    global selected_image, selected_prompt, latent, pipe, refiner, device, test_filepath
    global inpainter

    if inpainter:
        inpainter = None

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    image = Image.new("RGBA", (1024, 1024))
    root_layer = [l if l["label"] == "root" else {} for l in layers][0]
    root_layer_history_index = root_layer["currentLayerHistoryIndex"]
    root_bbox = root_layer["history"][root_layer_history_index]["bbox"]
    rx1, ry1, rx2, ry2 = root_bbox
    root_layer_width = rx2 - rx1
    root_layer_height = ry2 - ry1
    scale_x = 1024 / root_layer_width
    scale_y = 1024 / root_layer_height
    scale = min(scale_x, scale_y)
    margin_x = int((1024 - root_layer_width * scale) // 2)
    margin_y = int((1024 - root_layer_height * scale) // 2)
    print(f"margin_x: {margin_x} margin_y: {margin_y}")
    for layer in layers:

        history_index = layer["currentLayerHistoryIndex"]
        history = layer["history"]
        image_data = history[history_index]["imageData"]
        bbox = history[history_index]["bbox"]
        print(f"bbox: {bbox}")
        bbox = [int(n * scale) for n in bbox]
        print(f"bbox: {bbox}")
        # bbox = cast(tuple[int, int, int, int], tuple([bbox[i] + margin_x if i % 2 == 0 else bbox[i] + margin_y for i in range(len(bbox))]))
        print(f"bbox: {bbox}")
        b64 = extract_base64_data(image_data)
        image_bytes = base64.b64decode(b64)

        print(f"bbox: {bbox}, bbox width: {bbox[2] - bbox[0]} bbox height: {bbox[3] - bbox[1]}")

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        b64 = extract_base64_data(image_data)
        image_bytes = base64.b64decode(b64)


        layer_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        layer_image = layer_image.resize((w, h))
        print(f"bbox width: {bbox[2] - bbox[0]} bbox height: {bbox[3] - bbox[1]}")
        print(f"layer_image width: {layer_image.width} layer_image height: {layer_image.height}")

        image.paste(layer_image, box=tuple[int, int, int, int](bbox))

    image.save(os.path.join(os.getcwd(), f"{test_filepath}/img2img_original.png"))

    # width_scale = 1024 / image.width
    # height_scale = 1024 / image.height
    # scale = min(width_scale, height_scale)
    # image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))

    if refiner is None:
        await init_refiner()

    refiner = cast(StableDiffusionXLImg2ImgPipeline, refiner)

    await load_embeddings()

    image = image.convert("RGB")
    if remove_background:
        image = cast(Image.Image, remove(image, bgcolor=(127, 127, 127, 255)))
    image = ImageOps.pad(image, (1024, 1024), color=(0, 0, 0))

    print(f"image: {image}")

    denoising_start = 1.0 - strength

    output = cast(StableDiffusionXLImg2ImgPipeline, refiner)(
        prompt=prompt,
        image=image,
        # latents=latent,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        crop_coords_top_left=(0, 0),
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        output_type="pil",
        original_size=image.size,
        target_size=(1024, 1024),
        denoising_start=0.0,
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

    image_data = base64.b64encode(buffer.read()).decode('utf-8')



    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"image_data": image_data}


# class LayerFilter(ImageFilter.Filter):
#     def __init__(self, image: Image.Image):
#         self.image = image

#     def filter(self, image: Image.Image) -> ImagingCore:
#         filtered_image = Image.new(mode="RGBA", size=self.image.size)

async def inpaint(layers, prompt="", mask_data = [], strength=0.5, inference_steps=50, guidance_scale = 7.5, use_refiner=False, inpaint_refiner_ratio=0.85, inpaint_refiner_inference_steps=5,  inpaint_refiner_guidance_scale=7.5, negative_prompt=None, prompt_2=None, negative_prompt_2=None, refiner_prompt=None, refiner_negative_prompt=None, refiner_prompt_2=None, refiner_negative_prompt_2=None, new_layer=False, remove_background=False, callback_on_step_end=None):

    global inpainter, inpaint_image_count, buffer, inpaint_mask_count, inpaint_alpha_mask_count, image_count, refiner, pipe

    if pipe is not None:
        pipe.to("cpu")
        pipe = None

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if inpainter is None:
        await init_inpainter()

    await load_embeddings()

    image = Image.new("RGBA", (1024, 1024))
    root_layer = [l if l["label"] == "root" else {} for l in layers][0]
    root_layer_history_index = root_layer["currentLayerHistoryIndex"]
    root_bbox = root_layer["history"][root_layer_history_index]["bbox"]
    print(f"root_bbox: {root_bbox})")
    rx1, ry1, rx2, ry2 = root_bbox
    root_layer_width = rx2 - rx1
    root_layer_height = ry2 - ry1
    size = max(root_layer_width, root_layer_height)
    scale_x =  1024 / root_layer_width
    scale_y =  1024 / root_layer_height
    scale = min(scale_x, scale_y)
    print(f"scale: {scale}")
    margin_x = (size - root_layer_width) // 2
    margin_y = (size - root_layer_height) // 2

    print(f"margin_x: {margin_x} margin_y: {margin_y}")


    for layer in layers:

        history_index = layer["currentLayerHistoryIndex"]
        history = layer["history"]
        image_data = history[history_index]["imageData"]
        bbox = history[history_index]["bbox"]
        print(f"bbox: {bbox}")


        # bbox = cast(tuple[int, int, int, int], tuple([bbox[i] + margin_x if i % 2 == 0 else bbox[i] + margin_y for i in range(len(bbox))]))
        bbox = cast(tuple[int, int, int, int], tuple([ int(n * scale) for n in bbox]))
        print(f"bbox: {bbox})")

        b64 = extract_base64_data(image_data)
        image_bytes = base64.b64decode(b64)

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        layer_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        layer_image = layer_image.resize((w, h))
        image.paste(layer_image, box=bbox)

    print(f"image: {image}")

    image.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-composite-image.png"))
    # b64 = extract_base64_data(image)
    # image_bytes = base64.b64decode(b64)

    # image = Image.open(BytesIO(image_bytes)).convert("RGB")

    composite_mask = Image.new("L", image.size, 0)

    mask_count = 0

    mask_data = sorted(mask_data, key=lambda x: x["area"], reverse=True)
    composite_mask_bbox = [composite_mask.width, composite_mask.height, 0, 0]

    for data in mask_data:
        mask = data["mask"] if data["include"] == True else data["inverted_mask"] if data["exclude"] == True else None
        if mask is None:
            continue
        b64 = extract_base64_data(mask)
        image_bytes = base64.b64decode(b64)

        mask = Image.open(BytesIO(image_bytes))
        mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-original.png"))


        alpha_channel = mask.getchannel("A").convert("L")
        # alpha_channel.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step-{step}-alpha_channel.png"))

        # print(f"alpha_channel: {alpha_channel}")

        rgb = mask.convert("RGB")
        # rgb.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-rgb.png"))

        # width_scale = 1024 / mask.width
        # height_scale = 1024 / mask.height
        # scale = min(width_scale, height_scale)

        binary_mask = alpha_channel.point(lambda p: 255 if p > 0 else 0).convert("L")
        # print(f"binary_mask: {binary_mask}")
        # binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-binary.png"))

        # binary_mask = binary_mask.resize((int(math.floor(mask.width * scale)), int(math.floor(mask.height * scale))))
        # binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step-{step}-resized.png"))

        # binary_mask = ImageOps.pad(binary_mask, (1024, 1024), color=(0))
        print(f"binary_mask: {binary_mask}")
        # binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-{mask_count}-step--{step}-padded.png"))


        inverted_binary = ImageOps.invert(binary_mask)
        # inverted_binary.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-inverted-binary.png"))
        bbox = [int(n * scale) for n in data["bbox"]]
        bbox = cast(tuple[int, int, int, int], tuple([bbox[i] + margin_x if i % 2 == 0 else bbox[i] + margin_y for i in range(len(bbox))]))
        bbox = [int(n) for n in bbox]
        x1, y1, x2, y2 = bbox

        cx1, cy1, cx2, cy2 = composite_mask_bbox
        composite_mask_bbox = min(x1, cx1), min(y1, cy1), max(x2, cx2), max(y2, cy2)

        w, h = x2 - x1, y2 - y1
        binary_mask = binary_mask.resize((w, h))
        print(f"bbox: {bbox}")
        composite_mask.paste(binary_mask if data["include"] else ImageOps.invert(binary_mask), cast(tuple[int, int, int, int],tuple(bbox)), binary_mask)

        # composite_mask = ImageChops.lighter(composite_mask, binary_mask)
        # composite_mask = ImageChops.logical_or(composite_mask, binary_mask)

        # composite_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-composite.png"))
        print(f"composite_mask: {composite_mask}")

        mask_count += 1


    # composite_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-composite_binary.png"))

    composite_mask = composite_mask.convert("L")
    print(f"composite_mask: {composite_mask}")
    # composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/mask-{mask_count}-composite_L.png"))
    lut = [255 if i > 0 else 0 for i in range(256)]
    composite_mask = composite_mask.point(lut).convert('L')
    composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-composite-mask.png"))

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    # image = image.resize((int(math.floor(image.width * scale)), int(math.floor(image.height * scale))))
    composite_mask = composite_mask.resize((image.width, image.height))

    image = ImageOps.pad(image, (1024, 1024), color=127)
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
    # enhancer.enhance(0.5)

    composite_mask = composite_mask.copy()
    # composite_mask = composite_mask.filter(ImageFilter.GaussianBlur(3))

    composite_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-mask-final-input.png"))
    image.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-image-final-input.png"))

    assert image.mode == "RGB", f"Unexpected image mode: {image.mode}"
    assert composite_mask.mode == "L", f"Unexpected mask mode: {composite_mask.mode}"
    assert image.size == composite_mask.size, f"Size mismatch: {image.size} vs {composite_mask.size}"

    if remove_background:
        image = cast(Image.Image, remove(image, bgcolor=(127, 127, 127, 255)))

    image = image.convert("RGB")

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
        target_size=(1024, 1024),
        output_type="latent" if use_refiner else "pil",
        denoising_end=inpaint_refiner_ratio if use_refiner else 1.0,
        callback_on_step_end=callback_on_step_end).images[0]

    print(f"type(output): {type(output)}")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_refiner:

        if refiner is None:
            inpainter.to("cpu")
            inpainter = None
            await init_refiner()

        output = output.unsqueeze(0)
        output = cast(StableDiffusionXLImg2ImgPipeline, refiner)(
            prompt=refiner_prompt,
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
    #
    if new_layer:
        image.putalpha(composite_mask)
        output.putalpha(composite_mask)
        image_arr = np.asarray(image)
        output_arr = np.asarray(output)

        output_alpha = np.where(output_arr != image_arr, output_arr, image_arr * 0)
        output_alpha = Image.fromarray(output_alpha).convert("L")
        output.putalpha(output_alpha)
        output = output.crop(tuple[int, int, int, int](composite_mask_bbox))

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


async def tokenize(text: str):
    global tokenizer
    if tokenizer is None:
        await init_tokenizer()
        tokenizer = cast(CLIPTokenizer, tokenizer)

    return {"tokens": tokenizer.tokenize(text)}
