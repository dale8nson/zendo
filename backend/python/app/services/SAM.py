from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import SamModel, SamProcessor
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
from typing import cast
from pydantic.main import BaseModel
from typing import List
import os
import time
import math
import uuid

sam_model = None
sam_processor = None
predictor = None

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
    global mask_generator, sam_model, sam_processor, predictor

    sam_model_type = "vit_h"
    checkpoint = os.path.join(os.getcwd(),"../models/SAM/sam_vit_h_4b8939.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint).to(device)

    # predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")
    # sam_model = SamModel.from_pretrained(os.path.join(os.getcwd(),"../models/sam_model"))
    # sam_processor = SamProcessor.from_pretrained(os.path.join(os.getcwd(),"../models/sam_processor")


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

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    # inputs = sam_processor(image, input_points=[[[width, height]]])
    print(f"Completed in {time.time() - t:.2f} seconds")
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    j = 0
    mask_count = 1
    step = 1
    for mask in masks:

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

def depad(image: Image.Image, original_size: tuple[int, int]):
    w, h = original_size
    x1 = 0
    y1 = 0
    x2 = x1 + w
    y2 = y1 + h

    # crop left side
    image = image.crop((image.width / 2 - w / 2, image.height / 2 - h / 2, image.width / 2 + w / 2, image.height / 2 + h / 2))
    # crop right side
    # image = image.crop((w, 0,image.width, image.height))
    # # crop top
    # image = image.crop((0, 0,image.width, image.height / 2 - h / 2))
    # # crop bottom
    # image = image.crop((0, h, image.width, image.height))

    return image

async def generate_mask(image: str, bbox: List[int]):
    b64 = extract_base64_data(image)
    bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(bytes)).convert("RGB")
    enhancer = ImageEnhance.Contrast(img)
    enhancer.enhance(1.75)
    # img.save(os.path.join(os.getcwd(),"app/test_images/SAM-generate-mask-original.png"))
    x1, y1, x2, y2 = bbox

    cropped = img.crop((x1, y1, x2, y2))
    # scale_x = 512 / cropped.width
    # scale_y = 512 / cropped.height
    # scale = min(scale_x, scale_y)
    # cropped = cropped.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # cropped = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)), Image.BICUBIC)
    # padded = ImageOps.pad(cropped, (512, 512), color=(128, 128, 128))
    # cropped.save(os.path.join(os.getcwd(),"app/test_images/SAM-generate-mask-cropped.png"))

    # scale_x = 1024 / image.width
    # scale_y = 1024 / image.height
    # scale = min(scale_x, scale_y)
    # scaled = ImageOps.scale(cropped, scale)
    # scaled.save(os.path.join(os.getcwd(),"app/test_images/BLIP-generate-mask-scaled.png"))

    # bordered = ImageOps.expand(scaled, border=(math.floor(1024 - (scaled.width / 2)), math.floor(1024 - (scaled.height / 2))), fill=(255, 255, 255))
    # bordered.save(os.path.join(os.getcwd(),"app/test_images/BLIP-generate-mask-bordered.png"))

    buf = BytesIO()
    # padded.save(buf, format="png")
    cropped.save(buf, format="png")
    buf.seek(0)
    # cv_im = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
    # predictor.set_image(cv_im)
    # x1 = padded.width / 2 - cropped.width / 2
    # y1 = padded.height / 2 - cropped.height / 2
    # x2 = x1 + cropped.width
    # y2 = y1 + cropped.height
    # x1, y1, x2, y2 = bbox
    # box = np.array([x1, y1, x2, y2]).reshape(1, -1)
    # box = np.array(bbox).reshape(1, -1)

    # point_coords = np.array([[x1 + math.floor(i / 16 * (x2 - x1)), math.floor(j / 16 * (y2 - y1))] for i in range(0, 16) for j in range(0, 16)])
    # half_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
    # x0, y0 = half_point

    # density = 360 // 16
    # point_coords = np.array([[(half_point[0] + np.cos(density * i / 360 * np.pi / 180) // half_point[0]), (half_point[1] * density * i / 360 * np.pi / 180) // half_point[1]] for i in range(density)])

    #
    # point_coords = np.array([
    #     [(x1 + x2) // 2, (y1 + y2) // 2],  # center
    #     [x1 + 5, y1 + 5],                  # top-left (slightly inside)
    #     [x2 - 5, y1 + 5],                  # top-right
    #     [x1 + 5, y2 - 5],                  # bottom-left
    #     [x2 - 5, y2 - 5],
    # ])

    # point_labels = np.ones([len(point_coords)])

    # masks, scores, logits = predictor.predict(
    #     box=box,
    #     point_coords=point_coords,
    #     point_labels=point_labels,
    #     multimask_output=True,
    # )
    # mask_data = []
    # mask = masks[np.argmax(scores)]
    # for mask in masks:
    # data = {}
    # data["id"] = uuid.uuid4().hex
    # data["label"] = ""
    # data["active"] = False
    # data["include"] = False
    # data["exclude"] = False

    # index = 0
    # max_score = 0
    # for i in range(len(scores)):
    #     if scores[i] > max_score:
    #         max_score = scores[i]
    #         index = i

    # arr = mask
    # arr = arr.astype(np.uint8) * 255
    # alpha = Image.fromarray(arr, mode='L')
    # alpha.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-alpha.png"))
    # m = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
    # m.putalpha(alpha)
    # m.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-mask.png"))
    # m = depad(m, cropped.size)
    # m = m.resize((int(cropped.width / scale), int(cropped.height / scale)))

    # buf = BytesIO()
    # m.save(buf, format="PNG")
    # buf.seek(0)
    # b64 = base64.b64encode(buf.read()).decode('utf-8')
    # data["mask"] = b64

    # inverted_mask = Image.new('RGBA', alpha.size, (0, 0, 0, 255))
    # inverted_mask.putalpha(alpha)
    # inverted_mask.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-inverted-mask.png"))
    # inverted_mask = depad(inverted_mask, cropped.size)
    # inverted_mask = inverted_mask.resize((int(cropped.width / scale), int(cropped.height / scale)))

    # buf = BytesIO()
    # inverted_mask.save(buf, format="PNG")
    # buf.seek(0)
    # b64 = base64.b64encode(buf.read()).decode('utf-8')
    # data["inverted_mask"] = b64

    # active_mask = Image.new('RGBA', alpha.size, (247, 18, 224, 127))
    # active_mask.putalpha(alpha)
    # active_mask.save(os.path.join(os.getcwd(), f"app/test_images/SAM-mask-segmentation.png"))

    # active_mask = depad(active_mask, cropped.size)
    # active_mask = active_mask.resize((int(cropped.width / scale), int(cropped.height * scale)))

    # buf = BytesIO()
    # active_mask.save(buf, format="PNG")
    # buf.seek(0)
    # b64 = base64.b64encode(buf.read()).decode('utf-8')
    # data["segmentation"] = b64

    b64 = base64.b64encode(buf.read()).decode('utf-8')

        # mask_data.append(data)

    # return [data]
    return await generate_masks(image_data=b64, width=cropped.width, height=cropped.height)
