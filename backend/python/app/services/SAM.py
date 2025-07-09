from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import cv2
import base64
from app.utils.b64 import extract_base64_data
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from typing import cast
from pydantic.main import BaseModel
import os
import time


class MaskRequest(BaseModel):
    image: str

mask_generator = None

async def init_SAM():
    print("Initializing SAM...")
    global mask_generator

    sam_model_type = "vit_h"
    checkpoint = os.path.join(os.getcwd(),"../models/SAM/sam_vit_h_4b8939.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint).to(device)

    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

def dd(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')

    return [{"segmentation": b64, "bbox":[0, 0, 1024, 1024], "crop_box": [0, 0, 1024, 1024]}]

async def generate_masks(data_url: str):
    global mask_generator
    if mask_generator is None:
        await init_SAM()
    b64 = extract_base64_data(data_url)
    bytes = base64.b64decode(b64)
    image = Image.open(BytesIO(bytes))
    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    print(f"Image size: {image.width}x{image.height}, mode: {image.mode}, format: {image.format}")
    arr = np.asarray(image)
    print(f"Array shape: {arr.shape} Array dtype: {arr.dtype}")
    image = image.convert('RGB')
    arr = np.asarray(image)
    print(f"Array shape: {arr.shape} Array dtype: {arr.dtype}")
    print(f"Image size: {image.width}x{image.height}, mode: {image.mode}, format: {image.format}")

    array = np.asarray(image)

    print("Generating masks...")
    t = time.time()
    masks = cast(SamAutomaticMaskGenerator, mask_generator).generate(array)
    print(f"Completed in {time.time() - t:.2f} seconds")
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    j = 0
    for mask in masks:
        arr = mask["segmentation"]

        arr = arr.astype(np.uint8) * 255
        alpha = Image.fromarray(arr, mode='L')

        buf = BytesIO()
        alpha.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask["mask"] = b64

        img = Image.new('RGBA', alpha.size, (247, 18, 224, 255))
        img.putalpha(alpha)

        # img.save(os.path.join(os.getcwd(), f"app/test_images/mask_{j}.png"))
        # j += 1

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask["segmentation"] = b64

    return masks
