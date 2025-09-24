from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,  StableDiffusionXLControlNetPipeline, ControlNetModel, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType
import asyncio
import argparse
import random


from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import torch
import torchvision
import torchvision.transforms as T
from safetensors.torch import load_file
from safetensors import safe_open

from pydantic.main import BaseModel
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import os
import base64
from io import BytesIO
from typing import cast, List, Any, TypedDict, Union, Optional
import numpy as np
from rembg import remove
from datetime import datetime, timezone
import cv2
import re
import sys
from traceback import print_tb
import math
from tqdm.auto import tqdm
import json


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

# torch.cuda.is_available = is_available

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = None
tokenizer = None
tokenizer_2 = None
text_encoder = None
text_encoder_2 = None
vae = None
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
    guidance_scale: float
    negative_prompt: str
    prompt_2: str
    negative_prompt_2: str
    ip_adapter_image: str | List[Any] | None
    use_face_id: bool
    bbox: List[float] | None
    remove_background: bool
    use_ip_adapter_image: bool
    refiner_strength: float
    seed: int
    threshold1: int = 100
    threshold2: int = 200
    aperture_size: int = 3
    l2_gradient: bool = False

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

async def init_SDXL(model_path=os.path.join(os.getcwd(),"../models/sdxl-base-1.0")):
    print("Initializing SDXL pipeline...")
    global pipe, device, refiner, inpainter
    print("Device:", device)
    if pipe is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

# def load_tokens(path, pipeline):
#     global device, dtype
    
#     tok = pipeline.tokenizer_2 if '_2' in path else pipeline.tokenizer
#     if tok is None:
#         return
#     enc = pipeline.text_encoder_2 if '_2' in path else pipeline.text_encoder
#     if enc is None:
#         return
    
#     f = load_file(path, device=device.type)
#     for token in f.keys():
        
#         name = f'<{token}>' if not token.startswith('<') else token
#         tok.add_tokens(name)
#         enc.resize_token_embeddings(len(tok))
#         token_id = tok.convert_tokens_to_ids(name)
#         emb = f[token].to(device, dtype=dtype)
#         with torch.no_grad():
#             enc.get_input_embeddings().weight[token_id] = emb
    

def sanity_check(pipe, token):
    name = token if token.startswith("<") else f"<{token}>"
    id1 = pipe.tokenizer.convert_tokens_to_ids(name)
    id2 = pipe.tokenizer_2.convert_tokens_to_ids(name)
    assert id1 != pipe.tokenizer.unk_token_id, f"{name} missing in tokenizer (enc1)"
    assert id2 != pipe.tokenizer_2.unk_token_id, f"{name} missing in tokenizer_2"
    W1 = pipe.text_encoder.get_input_embeddings().weight
    W2 = pipe.text_encoder_2.get_input_embeddings().weight
    print(f"{name}: enc1 dim={W1.shape[1]} norm={float(W1[id1].norm()):.3f} | "
          f"enc2 dim={W2.shape[1]} norm={float(W2[id2].norm()):.3f}")

def embedding_row_and_norm(pipeline, token_str, enc2=False):
    tok = pipeline.tokenizer_2 if enc2 else pipeline.tokenizer
    enc = pipeline.text_encoder_2 if enc2 else pipeline.text_encoder
    tid = tok.convert_tokens_to_ids(token_str)
    w = enc.get_input_embeddings().weight.detach().cpu()
    return w[tid], w[tid].norm().item(), tid


def load_tokens(path, pipeline):
    """
    Load ALL token vectors stored as keys in a safetensors file into the
    correct tokenizer/encoder (enc1 or enc2), depending on filename suffix '_2'.
    """
    global device, dtype

    basename = os.path.basename(path)
    use_enc2 = ("_2" in basename)

    tok = pipeline.tokenizer_2 if use_enc2 else pipeline.tokenizer
    enc = pipeline.text_encoder_2 if use_enc2 else pipeline.text_encoder
    if tok is None or enc is None:
        return

    # Gather tokens and vectors from the file
    items = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # ignore possible non-embedding keys if present
        keys = [k for k in keys if k != "string_to_param" and not k.startswith("__")]
        for key in keys:
            name = key if key.startswith("<") else f"<{key}>"
            vec = f.get_tensor(key)  # shape [D] or [1, D]
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
            if vec.ndim != 1:
                raise ValueError(f"{path}:{key} must be 1D or 1xD, got {tuple(vec.shape)}")
            items.append((name, vec))

    if not items:
        return

    # Add only truly new tokens (once), then resize once
    vocab = tok.get_vocab()
    new_tokens = [name for (name, _) in items if name not in vocab]
    if new_tokens:
        tok.add_tokens(new_tokens)
        enc.resize_token_embeddings(len(tok))

    # Re-acquire embeddings after a possible resize
    emb = enc.get_input_embeddings()
    dim = emb.embedding_dim

    # Assign vectors with shape+dtype checks
    with torch.no_grad():
        for name, vec in items:
            tid = tok.convert_tokens_to_ids(name)
            if tid is None:
                raise RuntimeError(f"Token {name} unresolved in tokenizer after add()")
            if vec.shape[0] != dim:
                raise ValueError(
                    f"Dim mismatch for {name} from {basename}: got {vec.shape[0]}, expected {dim}"
                )
            emb.weight[tid].copy_(vec.to(device=device, dtype=emb.weight.dtype))
    print(f"Loaded {len(items)} token(s) from {basename} into {'enc2' if use_enc2 else 'enc1'}.")
# async def load_embeddings():
#     global pipe, refiner, inpainter, controlnet
    
#     path = os.path.join(os.getcwd(), "../models/user")

#     dir_list = [d for d in filter(lambda f: os.path.isdir(os.path.join(path,f)), os.listdir(path))]

#     print(f"dir_list: {dir_list}")
    
#     progress_bar = tqdm(range(0, len(dir_list)), initial=0, desc="Loading embeddings")

#     for sd in dir_list:
#         dir_path = os.path.join(path, sd)
#         progress_bar.set_description(f"Loading embeddings: {sd}")
        
#         files = [f for f in filter(lambda s: re.match(f"{sd}(_2)?\\.safetensors", s), os.listdir(dir_path))]


#         if len(files) < 2:
#             files = [f for f in filter(lambda n: re.match(f"{sd}(_2)?-steps-(\\d+).safetensors", n), dir_list)]

#             if len(files) < 2:
#                 continue
#             else:
#                 files = sorted(files, key=lambda f: int(f.split('-')[2].split('.')[0]))[-2:]

#         print(f"files: {files}")

#         for file in files:
#             progress_bar.set_description(f"Loading {file}")
            
#             tensor_path = os.path.join(dir_path, file)

#             try:
#                 if pipe is not None and isinstance(pipe, StableDiffusionXLPipeline):
                    
                    
#                     # pipe.unload_textual_inversion()
#                     load_tokens(tensor_path, pipe)
                    
#                     # pipe.load_textual_inversion(tensor_path)
#             except FileNotFoundError as e:
#                 print(e)
#             except Exception as e:
#                 print(e)

#             try:
#                 if refiner is not None and isinstance(refiner, StableDiffusionXLImg2ImgPipeline):
#                     # refiner.unload_textual_inversion()
#                     # refiner.load_textual_inversion(tensor_path)
#                     load_tokens(tensor_path, refiner)
#             except FileNotFoundError as e:
#                 print(e)
#             except Exception as e:
#                 print(e)

#             try:
#                 if inpainter is not None and isinstance(inpainter, StableDiffusionXLInpaintPipeline):
#                     # inpainter.unload_textual_inversion()
#                     # inpainter.load_textual_inversion(tensor_path)
#                     load_tokens(tensor_path, inpainter)
#             except FileNotFoundError as e:
#                 print(e)
#             except Exception as e:
#                 print(e)
#             try:
#                 if controlnet is not None and isinstance(controlnet, StableDiffusionXLControlNetPipeline):
#                     # controlnet.unload_textual_inversion()
#                     load_tokens(tensor_path, controlnet)
#                     # controlnet.load_textual_inversion(tensor_path)
#             except FileNotFoundError as e:
#                 print(e)
#             except Exception as e:
#                 print(e)
#             progress_bar.update(1 / len(files))

#         progress_bar.update(1)


async def load_embeddings(base=os.path.join(os.getcwd(), "../models/user")):
    global pipe, refiner, inpainter, controlnet

    collections = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    print(f"collections: {collections}")

    pbar = tqdm(total=len(collections), desc="Loading embeddings")
    for coll in collections:
        dir_path = os.path.join(base, coll)

        # Prefer exact names: <collection>.safetensors and <collection>_2.safetensors
        exact = [f for f in os.listdir(dir_path)
                 if re.fullmatch(rf"{re.escape(coll)}(_2)?\.safetensors", f)]

        files = exact[:]
        if len(files) < 2:
            # Fallback: pick the latest step per side inside THIS directory
            step_candidates = [f for f in os.listdir(dir_path)
                               if re.fullmatch(rf"{re.escape(coll)}(_2)?-steps-(\d+)\.safetensors", f)]

            def stepnum(x):
                m = re.search(r"-steps-(\d+)\.safetensors$", x)
                return int(m.group(1)) if m else -1

            main = sorted([c for c in step_candidates if "_2" not in c], key=stepnum)
            aux  = sorted([c for c in step_candidates if "_2"     in c], key=stepnum)

            files = []
            if main:
                files.append(main[-1])  # latest main
            if aux:
                files.append(aux[-1])   # latest _2

        if len(files) < 2:
            print(f"Skipping {coll}: found files={files}")
            pbar.update(1)
            continue

        # Load both sides into all active pipelines, for ALL tokens in those files
        for fname in files:
            tensor_path = os.path.join(dir_path, fname)
            for p in (pipe, refiner, inpainter, controlnet):
                try:
                    if p is not None:
                        load_tokens(tensor_path, p)
                except Exception as e:
                    print(f"Failed {tensor_path} on {type(p).__name__}: {e}")

        pbar.update(1)

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

def layers_to_pil(data: str | List) -> Image.Image:

    image = Image.new("RGBA", (1024, 1024))
    if isinstance(data, List):

        root_layer = [l if l["label"] == "root" else {} for l in data][0]
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

        for layer in data:
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

        print(f'image: {image}')
        image.save(os.path.join(os.getcwd(),f'app/test_images/generate-layered-image-{datetime.now(timezone.utc)}.png'))

    elif isinstance(data, str):
        b64 = extract_base64_data(data)
        image_bytes = base64.b64decode(b64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

    scale_x = 1024 / image.width
    scale_y = 1024 / image.height
    scale = min(scale_x, scale_y)
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    image = ImageOps.pad(image, (1024, 1024))

    return image


def pil2pixels(image: Image.Image) -> torch.Tensor:
    image = image.convert('RGB')
    arr = np.asarray(image).astype(np.uint8)
    # arr = (arr / 127.5 - 1.0).astype(np.float32)
    arr = arr.astype(np.float32)
    pixel_values = torch.from_numpy(arr).to(device)

    # pixel_values = pixel_values * 2 - 1

    return pixel_values

def generate_latent(image: Image.Image, pipeline: DiffusionPipeline) -> torch.Tensor:

    arr = np.asarray(image).astype(np.uint8)
    print(f'arr: {arr}')
    arr = (arr / 127.5 - 1.0).astype(np.float32)
    pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
    print(f'pixel_values.shape: {pixel_values.shape}')
    latents = pipeline.vae.encode(pixel_values).latent_dist.sample()


    return latents

def add_noise(latent, pipeline, steps):

    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    # timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, latent.shape[0], device=latent.device)
    # timesteps = timesteps.long()
    latent_timestep = timesteps[:1]
    shape = latent.shape
    noise = torch.randn(shape, device=device, dtype=torch.float32)

    noisy_latent = pipeline.scheduler.add_noise(latent, noise, latent_timestep)

    return noisy_latent

async def generate(prompt, iterations, guidance_scale, negative_prompt, prompt_2, negative_prompt_2, ip_adapter_image=None, use_face_id=False, bbox=None, remove_background=False, use_ip_adapter_image=False, refiner_strength=0.2, seed=None, threshold1=100, threshold2=200, aperture_size=3, l2_gradient=False) -> dict:

    global selected_image, pipe, selected_prompt, latent, prompt_embeds, controlnet, negative_prompt_embeds, inpainter, refiner
    
    if seed is None:
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    
    prompt_2 = prompt if prompt_2 == "" or prompt_2 is None else prompt_2
    if negative_prompt_2 == "": negative_prompt_2 = negative_prompt
    
    try:
        if refiner is None:
            await init_refiner()
            refiner = cast(StableDiffusionXLImg2ImgPipeline, refiner)

        if inpainter is not None:
            inpainter.to("cpu")
            inpainter = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if pipe is None:
            await init_SDXL()
            pipe = cast(StableDiffusionXLPipeline, pipe)

        if ip_adapter_image is not None:

            if controlnet is None:
                await init_controlnet()
                controlnet = cast(StableDiffusionXLControlNetPipeline, controlnet)

        latent = None
        
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],   # SDXL: pooled for encoder 2
        )

        if ip_adapter_image is not None:
            if use_face_id and pipe is not None:
                pipe.load_ip_adapter(
                    '../models/IP-Adapter-FaceID',
                    subfolder=".",
                    weight_name="pytorch_model.bin",
                    use_safetensors=True,
                    slice_size=1
                )

            print(f"type(ip_adapter_image): {type(ip_adapter_image)}")
            if isinstance(ip_adapter_image, list):
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

                    image.paste(layer_image, box=tuple[int,int, int, int](bbox))

                ip_adapter_image = image

            else:
                b64 = extract_base64_data(ip_adapter_image)
                image_bytes = base64.b64decode(b64)

                ip_adapter_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            width_scale = 1024 / ip_adapter_image.width
            height_scale = 1024 / ip_adapter_image.height
            scale = min(width_scale, height_scale)

            ip_adapter_image = ip_adapter_image.resize((int(ip_adapter_image.width * scale), int(ip_adapter_image.height * scale)))

            ip_adapter_image = ImageOps.pad(ip_adapter_image, (1024, 1024), color=(0, 0, 0))

            if remove_background:
                try:
                    with torch.no_grad():
                        foreground = remove(ip_adapter_image, bgcolor=(0, 0, 0, 0))
                        foreground.save(os.path.join(cwd, 'app/test_images/generate-foreground.png'))
                        alpha = foreground.getchannel('A')
                        alpha.save(os.path.join(cwd, 'app/test_images/generate-alpha_channel.png'))
                        white_bg = Image.new(mode='RGB', size=alpha.size, color=(255, 255, 255))
                        black_bg = Image.new(mode='RGB', size=alpha.size, color=(0, 0, 0))

                        alpha = ImageChops.composite(white_bg, black_bg, alpha)
                        alpha.save(os.path.join(cwd, 'app/test_images/generate-alpha_rgb.png'))
                        # alpha = pil2pixels(alpha).to(torch.device('cpu'))
                        # alpha = torch.broadcast_to(alpha, (1, 3, 1024, 1024))
                        # alpha_latent = generate_latent(alpha, pipe) * pipe.vae.config.scaling_factor

                        foreground = foreground.convert('RGB')
                        fg_latent = generate_latent(foreground, pipe)  * pipe.vae.config.scaling_factor

                        # print(f'alpha_mask.shape: {alpha_mask.shape}')
                        # print(f'alpha_latent: {alpha_latent}')
                        # fg_latent = pipe.vae.config.scaling_factor * fg_latent

                        bg_latent = torch.randn_like(fg_latent)
                        print(f'bg_latent.shape: {bg_latent.shape}')

                        bg_latent = (bg_latent + 1) * 127.5
                        decoded = pipe.vae.decode(bg_latent).sample.squeeze(0).permute(1, 2, 0).cpu().to(dtype=torch.uint8)

                        print(f'decoded.shape: {decoded.shape}')

                        bg = Image.fromarray(np.asarray(decoded))
                        bg.save(os.path.join(cwd, 'app/test_images/generate-latent_image.png'))

                        init_timestep = min(int(math.ceil(iterations * 0.3)), iterations)
                        t_start = max(iterations - init_timestep, 0)
                        timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]

                        # latent = pipe.scheduler.add_noise(latent, bg_latent, timesteps[:1])
                        #
                        buf = BytesIO(alpha.tobytes())

                        alpha_tensor = torch.frombuffer(buf, dtype=torch.uint8).permute(2,0,1).unsqueeze(0).to(dtype=torch.float32)

                        alpha_latent = generate_latent(alpha, pipe) * pipe.vae.config.scaling_factor
                        alpha_latent = torch.tile(alpha_tensor, fg_latent.shape)

                        latent = torch.where(alpha_latent == 0, bg_latent, fg_latent).to(device)

                        latent_image = pipe.vae.decode(latent).sample.squeeze(0).permute((1,2,0)).cpu().detach().numpy().astype(np.uint8)
                        latent_image = Image.fromarray(latent_image).convert('RGB')
                        latent_image.save(os.path.join(cwd, 'app/test_images/generate-latent_image.png'))

                except Exception as e:
                    print(e, print_tb(e.__traceback__))
                    return {'status': repr(sys.exception())}

            if use_ip_adapter_image:

                ip_adapter_image = np.asarray(ip_adapter_image)
                ip_adapter_image = cv2.Canny(ip_adapter_image, threshold1, threshold2)
                if l2_gradient:
                    ip_adapter_image = cv2.Canny(ip_adapter_image, threshold1, threshold2, L2gradient=True)
                ip_adapter_image = cv2.Canny(ip_adapter_image, threshold1, threshold2, apertureSize=aperture_size)
                ip_adapter_image = ip_adapter_image[:, :, None]
                ip_adapter_image = np.concatenate([ip_adapter_image, ip_adapter_image, ip_adapter_image], axis=2)

                ip_adapter_image = Image.fromarray(ip_adapter_image)


            if not use_ip_adapter_image and latent is None:
                # pil = layers_to_pil(ip_adapter_image)
                pixels = pil2pixels(ip_adapter_image)
                latent = pipe.vae.encode(pixels).latent_dist.sample()
                print(f'latent.shape: {latent.shape}')
                latent *= pipe.vae.config.scaling_factor
                print(f'latent.shape: {latent.shape}')
                # latent = add_noise(latent, pipe, int(math.ceil(0.1 * iterations)))
                init_timestep = min(int(iterations * 0.3), iterations)
                t_start = max(iterations - init_timestep, 0)
                timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]
                latent = pipe.scheduler.add_noise(latent, pipe, timesteps[:1])
                print(f'latent.shape: {latent.shape}')

            # latent = pipe.vae.encode(ip_adapter_image).latent_dist.sample()

            # ip_adapter_image = torch.where(alpha_tensor == 0, bg_latent, ip_adapter_image)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selected_prompt = prompt
        
        tok_vocab_size = pipe.tokenizer.total_vocab_size
        tok2_vocab_size = pipe.tokenizer_2.total_vocab_size

        try:
            await load_embeddings()
            
            tok = pipe.tokenizer
            tok2 = pipe.tokenizer_2
            
            ids1 = tok(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            ids2 = tok2(prompt_2 or prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            
            m1 = torch.tensor([1 if id in tok.get_added_vocab().values() and id not in tok.all_special_ids else 0])
            m2 = torch.tensor([1 if id in tok2.get_added_vocab().values() and id not in tok2.all_special_ids else 0])


            out1 = pipe.text_encoder(ids1, output_hidden_states=True)
            out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)

            hs1 = out1.hidden_states[-2]            # [B,L,768]
            hs2 = out2.hidden_states[-2]            # [B,L,1280]
            pooled = out2[0]
                        

            alpha = 3.0  # try 1.2–3.0; >4 often causes artifacts/dominance
            
            hs1[m1] *= alpha
            
            hs2[m2] *= alpha
                

            prompt_embeds = torch.cat([hs1, hs2], dim=-1)
            
            def show_positions(pipe, prompt, token):
                tid1 = pipe.tokenizer.convert_tokens_to_ids(token)
                tid2 = pipe.tokenizer_2.convert_tokens_to_ids(token)
                ids1 = pipe.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).input_ids[0]
                ids2 = pipe.tokenizer_2(prompt, return_tensors="pt", padding=False, truncation=True).input_ids[0]
                pos1 = (ids1 == tid1).nonzero(as_tuple=True)[0].tolist()
                pos2 = (ids2 == tid2).nonzero(as_tuple=True)[0].tolist()
                print("enc1 positions:", pos1, "len:", ids1.shape[0])
                print("enc2 positions:", pos2, "len:", ids2.shape[0])

            # prompt = f"{t}, a photo of ..."   # put token FIRST during testing
            # prompt_2 = None                     # let SDXL copy prompt → prompt_2
            # show_positions(pipe, prompt, t)
            
            # use_ip_adapter_image = False
            if use_ip_adapter_image:
                image = controlnet(prompt=prompt, negative_prompt=negative_prompt, prompt_2=prompt_2, negative_prompt_2=negative_prompt_2, image=ip_adapter_image,
                latents=latent,
                denoising_start=0.0,
                guidance_scale=guidance_scale, num_inference_steps=iterations,
                output_type='latent' if refiner_strength > 0 else 'pil',
                denoising_end=1.0 - refiner_strength,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
                controlnet_conditioning_scale=0.65).images[0]

                # print(f'image.shape: {image.shape}')

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:

                image = pipe(prompt=None, prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds = pooled,
                    negative_prompt=negative_prompt, num_inference_steps=iterations, guidance_scale=guidance_scale, prompt_2=None,
                    latents=latent,
                    negative_prompt_2=negative_prompt_2, ip_adapter_image=ip_adapter_image if use_ip_adapter_image else None, width=1024,
                    denoising_start=0.0,
                    height=1024, denoising_end=1.0 - refiner_strength,
                    generator=torch.Generator(device=pipe.device).manual_seed(seed),
                    output_type='latent' if refiner_strength > 0 else 'pil').images[0]

                # print(f'image.shape: {image.shape}')

            # print(f'type(image):{type(image)} image.shape: {image.shape} image.dtype: {image.dtype} image.min(): {image.min()} image.max(): {image.max()}')
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0)
            
            if refiner_strength > 0:

                image = refiner( prompt=prompt, num_inference_steps=iterations, denoising_start=1.0 - refiner_strength, image=image).images[0]

        except KeyboardInterrupt as e:

            raise e

        if use_face_id and pipe is not None:
            pipe.unload_ip_adapter()

        print(f"image: {image}")
        os.makedirs(os.path.join(os.getcwd(), 'app/test_images'), exist_ok=True)
        image.save(os.path.join(os.getcwd(), f"app/test_images/generated-image-{datetime.now(timezone.utc)}.png"))

    except Exception as e:
        print(e, print_tb(e.__traceback__))
        return {'image': ''}

    buffer = BytesIO()
    selected_image = image
    selected_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": image_base64}

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

    os.makedirs(os.path.join(os.getcwd(), 'app/test_images'), exist_ok=True)
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

    g = torch.Generator(device=pipe.device).manual_seed(1234)

    common = dict(
        num_inference_steps=30,
        guidance_scale=7.0,
        width=1024, height=1024,
        latents=None,                 # fresh noise
        ip_adapter_image=None,
        negative_prompt=None,
        prompt_2=None,                # CRUCIAL: None mirrors to enc2; "" empties it
        negative_prompt_2=None,
        output_type="latent",
        generator=g,
    )

    # img_wo = pipe(prompt="a portrait", **common).images[0]
    # img_wo.save(os.path.join(os.getcwd(), f"{test_filepath}/img_wo.png"))
    # g = torch.Generator(device=pipe.device).manual_seed(1234)  # reset
    # img_w  = pipe(prompt="a portrait <your_token>", **common).images[0]
    # img_w.save(os.path.join(os.getcwd(), f"{test_filepath}/img_w.png"))

    # delta = (img_w - img_wo).abs().mean().item()
    # print("Mean |Δ| in latent space:", delta)

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

        # rgb.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-rgb.png"))

        # width_scale = 1024 / mask.width
        # height_scale = 1024 / mask.height
        # scale = min(width_scale, height_scale)

        binary_mask = alpha_channel.point(lambda p: 255 if p > 0 else 0).convert("L")
        # print(f"binary_mask: {binary_mask}")
        binary_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/inpaint-mask-{mask_count}-binary.png"))

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


    composite_mask.save(os.path.join(os.getcwd(), f"{test_filepath}/mask-composite_binary.png"))

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

        # output_alpha = np.where(output_arr != image_arr, output_arr, image_arr * 0)
        # output_alpha = Image.fromarray(output_alpha).convert("L")
        # output.putalpha(output_alpha)
        output = output.crop(tuple[int, int, int, int](composite_mask_bbox))

    output.save(os.path.join(os.getcwd(),f"{test_filepath}/inpaint-image-final-output.png"))
    print(f"image.size: {output.size}")
    # alpha_arr = np.ones((output.height, output.width)).astype(np.uint8) * alpha
    # alpha_mask = Image.fromarray(alpha_arr, mode="L")
    # alpha_mask.save(os.path.join(os.getcwd(),f"{test_filepath}/alpha_mask.png"))
    # output = output.convert("RGBA")
    # output.putalpha(alpha_mask)
    # new_image.save(os.path.join(os.getcwd(),f"{test_filepath}/new_image_with_alpha.png"))

    os.makedirs(os.path.join(os.getcwd(), 'app/test_images'), exist_ok=True)
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

    return {"image_data": b64, "bbox": composite_mask_bbox}


async def tokenize(text: str):
    global tokenizer
    if tokenizer is None:
        await init_tokenizer()
        tokenizer = cast(CLIPTokenizer, tokenizer)

    return {"tokens": tokenizer.tokenize(text)}

@torch.no_grad()
async def encode_decode(data: str):
    # device = torch.device('cpu')
    image = layers_to_pil(data)
    vae = AutoencoderKL.from_pretrained('../models/sdxl-base-1.0', subfolder='vae').to(device)
    vae.eval()
    pixel_values = pil2pixels(image).to(device, dtype=torch.float32)


    encoded = vae.encode(pixel_values).latent_dist
    latents = encoded.mean
    # latents = latents * vae.config.scaling_factor
    decoded = vae.decode(latents.to(device, dtype=torch.float32)).sample

    delta = (decoded - pixel_values).mean().item()     # brightness bias
    mae   = (decoded - pixel_values).abs().mean().item()
    print(delta, mae)

    decoded = (decoded.clamp(-1,1) + 1) / 2

    pil = torchvision.transforms.functional.to_pil_image(decoded.squeeze(0), mode='RGB').convert('RGBA')

    print(f'pil: {pil}')

    buf = BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode('utf-8')
    return image_data


async def test():
    global pipe
    
    await init_SDXL()
    
    pipe = cast(StableDiffusionXLPipeline, pipe)
    
    await load_embeddings()
    
    t = "<vpl>"  # exact string you trained/loaded
    base = "a photograph of a woman, outside, natural lighting"
    prompt   = f"{t}, {base}"
    prompt_2 = f"{t}, {base}"  # SDXL’s pooled branch (very important for style)

    neg = ""  # keep empty for the test
    
    ids1 = pipe.tokenizer(prompt, return_tensors="pt").input_ids
    ids2 = pipe.tokenizer_2(prompt_2, return_tensors="pt").input_ids
    tid1 = pipe.tokenizer.convert_tokens_to_ids(t)
    tid2 = pipe.tokenizer_2.convert_tokens_to_ids(t)
    print("token present:", (ids1 == tid1).any().item(), (ids2 == tid2).any().item())

    imgA = pipe(
        prompt=base, prompt_2=base,
        negative_prompt=neg, negative_prompt_2=neg,
        num_inference_steps=30, guidance_scale=7.5,
        width=1024, height=1024,
        output_type="pil",               # decode to image for a fair A/B
        generator=torch.Generator(device=pipe.device).manual_seed(1234),
    ).images[0]
    
    imgA.save('./app/test_images/imgA.png')

    imgB = pipe(
        prompt=prompt, prompt_2=prompt_2,
        negative_prompt=neg, negative_prompt_2=neg,
        num_inference_steps=30, guidance_scale=7.5,
        width=1024, height=1024,
        output_type="pil",
        generator=torch.Generator(device=pipe.device).manual_seed(1234),
    ).images[0]
    
    imgB.save('./app/test_images/imgB.png')
    
    
async def validate():
    
    global pipe
    
    await init_SDXL()
    
    pipe = cast(StableDiffusionXLPipeline, pipe)
    
    if pipe is None: return
    
    await load_embeddings()
    
    captions = json.load(open('../models/user/nsfw/captions.json'))
    
    filenames = [f for f in captions.keys() if captions[f]['token'] == '<vpl>']
    
    for filename in filenames:
        image = pipe(captions[filename]['caption'], num_inference_steps=30, guidance_scale=5).images[0]
        
        image.save(f'../validation/{captions[filename]['token']}-{datetime.now(timezone.utc)}-{captions[filename]['caption'].replace(' ', '_')[:15]}.png')
    

async def main():
    global pipe
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-p',help='prompt' , type=str)
    parser.add_argument('-i', type=int, help='iterations', default=15)
    parser.add_argument('-g', type=float, help='guidance scale', default=7.5)
    parser.add_argument('-np', help='negative prompt', type=str, default=None)
    parser.add_argument('-p2' , help='prompt 2', type=str, default=None)
    parser.add_argument('-np2', help='negative prompt 2', type=str, default=None)
    parser.add_argument('-s', help='manual random seed', type=int, default=random.randint(0, 0xFFFFFFFFFFFFFFFF))
    parser.add_argument('-d', type=str, help='output directory', default='.')
    parser.add_argument('-f', type=str, help='filename', default=f'output-{datetime.now(timezone.utc)}.png')
    
    args = parser.parse_args()
    
    if pipe is None:
        await init_SDXL(model_path='../../../models/sdxl-base-1.0')
        pipe = cast(StableDiffusionXLPipeline, pipe)
    
    await load_embeddings('../../../models/user')
    
    tok = pipe.tokenizer
    tok2 = pipe.tokenizer_2
    
    ids1 = tok(args.p, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    ids2 = tok2(args.p2 or args.p, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    
    m1 = torch.tensor([1 if id in list(tok.get_added_vocab().values()) and id not in tok.all_special_ids else 0])
    m2 = torch.tensor([1 if id in list(tok2.get_added_vocab().values()) and id not in tok2.all_special_ids else 0])

    out1 = pipe.text_encoder(ids1, output_hidden_states=True)
    out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)

    hs1 = out1.hidden_states[-2]            # [B,L,768]
    hs2 = out2.hidden_states[-2]            # [B,L,1280]
    pooled = out2[0]
                

    alpha = 3.0  # try 1.2–3.0; >4 often causes artifacts/dominance
    
    out1 = hs1[m1] * alpha
    
    out2 = hs2[m2] * alpha
        

    prompt_embeds = torch.cat([hs1, hs2], dim=-1)
    
    g = torch.Generator(device).manual_seed(args.s)
    
    # image = pipe(prompt=args.p, num_inference_steps=args.i, prompt_embeds=prompt_embeds, guidance_scale=args.g, negative_prompt=args.np, prompt_2=args.p2, negative_prompt_2=args.np2,  generator=g, output_type='pil').images[0]
    
    image = pipe(num_inference_steps=args.i, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, guidance_scale=args.g, negative_prompt=args.np, prompt_2=args.p2, negative_prompt_2=args.np2,  generator=g, output_type='pil').images[0]
    
    
    image.save(os.path.join(args.d, f'{args.f}.png'))
    

if __name__ == '__main__': asyncio.run(main())
