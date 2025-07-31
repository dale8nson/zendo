from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from typing import cast
from pydantic.main import BaseModel
from typing import List
import os
import time
import math
import uuid


class MaskRequest(BaseModel):
    image: str

mask_generator = None

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

async def init_SAM():
    print("Initializing SAM...")
    global mask_generator

    sam_model_type = "vit_h"
    checkpoint = os.path.join(os.getcwd(),"../models/SAM/sam_vit_h_4b8939.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint, weights_only=False).to(device)

    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

def dd(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')

    return [{"segmentation": b64, "bbox":[0, 0, 1024, 1024], "crop_box": [0, 0, 1024, 1024]}]

async def generate_masks(image_data: str, width: int = 1024, height: int = 1024):
    global mask_generator
    if mask_generator is None:
        await init_SAM()

    b64 = extract_base64_data(image_data)
    bytes = base64.b64decode(b64)
    image = Image.open(BytesIO(bytes)).convert("RGB")
    # print(f"image: {image}")
    image.save(os.path.join(os.getcwd(), "app/test_images/SAM_input_img.png"))
    # width_scale = 1024 / image.width
    # height_scale = 1024 / image.height
    # scale = min(width_scale, height_scale)
    # image = image.resize((int(image.width * scale), int(image.height * scale)))

    # print(f"Image size: {image.width}x{image.height}, mode: {image.mode}, format: {image.format}")
    # arr = np.asarray(image)
    # print(f"Array shape: {arr.shape} Array dtype: {arr.dtype}")
    # image = image.convert('RGB')
    # arr = np.asarray(image)
    # print(f"Array shape: {arr.shape} Array dtype: {arr.dtype}")
    # print(f"Image size: {image.width}x{image.height}, mode: {image.mode}, format: {image.format}")

    array = np.asarray(image)

    print("Generating masks...")
    t = time.time()
    masks = cast(SamAutomaticMaskGenerator, mask_generator).generate(array)
    print(f"Completed in {time.time() - t:.2f} seconds")
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    j = 0
    mask_count = 1
    step = 1
    for mask in masks:
        mask["id"] = j

        mask["id"] = uuid.uuid4().hex
        mask["active"] = False
        mask["include"] = False
        mask["exclude"] = False
        arr = mask["segmentation"]

        arr = arr.astype(np.uint8) * 255
        alpha = Image.fromarray(arr, mode='L')
        alpha.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{j}-alpha.png"))
        # step += 1
        m = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
        # m.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{mask_count}-step-{step}.png"))
        # step += 1
        m.putalpha(alpha)
        m.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{j}-mask.png"))
        # step += 1
        buf = BytesIO()
        m.save(buf, format="PNG")
        # m.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{mask_count}-step-{step}-mask.png"))
        # step += 1
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask["mask"] = b64

        inverted_mask = Image.new('RGBA', alpha.size, (0, 0, 0, 255))
        inverted_mask.putalpha(alpha)
        inverted_mask.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{j}-inverted-mask.png"))
        buf = BytesIO()
        inverted_mask.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask["inverted_mask"] = b64

        img = Image.new('RGBA', alpha.size, (247, 18, 224, 127))
        # img.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{mask_count}-step-{step}.png"))
        # step += 1
        img.putalpha(alpha)
        img.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-{j}-segmentation.png"))
        step += 1

        # img.save(os.path.join(os.getcwd(), f"app/test_images/mask_{j}.png"))
        # j += 1

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask["segmentation"] = b64

        mask["label"] = ""

        mask["active"] = False
        mask["include"] = False
        mask["exclude"] = False

        mask_count += 1
        step = 1
        j += 1


    return masks

def generate_mask(image: str, bbox: List[int]):
    b64 = extract_base64_data(image)
    bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(bytes)).convert("RGB")
    # img.save(os.path.join(os.getcwd(),"app/test_images/SAM-generate-mask-original.png"))
    x1, y1, x2, y2 = bbox

    cropped = img.crop((x1, y1, x2, y2))
    # cropped.save(os.path.join(os.getcwd(),"app/test_images/SAM-generate-mask-cropped.png"))

    # scale_x = 1024 / image.width
    # scale_y = 1024 / image.height
    # scale = min(scale_x, scale_y)
    # scaled = ImageOps.scale(cropped, scale)
    # scaled.save(os.path.join(os.getcwd(),"app/test_images/BLIP-generate-mask-scaled.png"))

    # bordered = ImageOps.expand(scaled, border=(math.floor(1024 - (scaled.width / 2)), math.floor(1024 - (scaled.height / 2))), fill=(255, 255, 255))
    # bordered.save(os.path.join(os.getcwd(),"app/test_images/BLIP-generate-mask-bordered.png"))

    buf = BytesIO()
    cropped.save(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return generate_masks(image_data=b64, width=cropped.width, height=cropped.height)
