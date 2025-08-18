import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms
import safetensors
from safetensors.torch import save_file, load_file
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel, get_scheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from pydantic.main import BaseModel
from typing import List, Tuple, Dict, cast
from PIL import Image, ImageEnhance, ImageChops, ImageOps, ImageFilter
import base64
from io import BytesIO
import os
import random
import json
import uuid
import math
import numpy as np
from tqdm.auto import tqdm
import re
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from app.services.SDXL import get_pipe, get_refiner, get_inpainter, get_controlnet, init_SDXL
from app.services.textual_inversion.textual_inversion_sdxl import TextualInversionDataset
import inspect
import gc
from datetime import datetime, timezone

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

class MaskData(BaseModel):
    image_data: str
    bbox: tuple[int, int, int, int]
    id: str
    segmentation: str
    area: int
    predicted_iou: float
    point_coords: List[int]
    stability_score: float
    crop_box: tuple[int, int, int, int]
    mask: str
    inverted_mask: str
    label: str
    active: bool
    include: bool
    exclude: bool
    canvas_box: List[int]

class DatasetPostRequest(BaseModel):
    image_data: str
    bbox: List[int]
    collection: str
    token: str
    caption: str


pipe = None
refiner = None
inpainter = None

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

def to_b64(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf8")
    return b64

step = 1

def resize_image(image, size=1024):
    scale_x = size / image.width
    scale_y = size / image.height
    scale = min(scale_x, scale_y)
    image = image.resize((math.floor(image.width * scale), math.floor(image.height * scale)))

    return image

def generate_images(image, bg, bbox, mask, size=1024):
    global step
    size = 1024

    image.save(os.path.join(os.getcwd(), "app/test_images/dataset-image-original.png"))
    mask.save(os.path.join(os.getcwd(), "app/test_images/dataset-mask-original.png"))
    bg.save(os.path.join(os.getcwd(), "app/test_images/dataset-bg-original.png"))


    print(f"image: {image}")
    print(f"bg: {bg}")
    print(f"bbox: {bbox}")
    print(f"mask: {mask}")

    full_mask = Image.new("L", (1024, 1024), color=0)

    positioned = bg.copy()
    cropped_mask = mask.crop(bbox)
    cropped_mask.save(os.path.join(os.getcwd(), f"app/test_images/dataset-cropped-mask-{step}.png"))


    x = math.floor(size / 2 - cropped_mask.width / 2)
    y = math.floor(size / 2 - cropped_mask.height / 2)
    center_bbox = (x, y, x + cropped_mask.width, y + cropped_mask.height)
    print(f"image: {image}")
    print(f"center_bbox: {center_bbox}")
    print(f"mask: {mask}")
    print(f"full_mask: {full_mask}")

    cropped_image = image.crop(bbox)
    cropped_image.save(os.path.join(os.getcwd(), "app/test_images/dataset-cropped-image.png"))

    print(f"cropped_image: {cropped_image}")
    print(f"cropped_mask: {cropped_mask}")
    positioned.paste(cropped_image, box=bbox, mask=cropped_mask)

    positioned.save(os.path.join(os.getcwd(), "app/test_images/dataset-positioned.png"))
    step += 1

    centered = bg.copy()

    x1, y1 = int(centered.width / 2 - cropped_image.width / 2), int(centered.height / 2 - cropped_image.height / 2)
    x2, y2 = x1 + cropped_image.width, y1 + cropped_image.height

    centered.paste(cropped_image, box=center_bbox, mask=cropped_mask)
    centered.save(os.path.join(os.getcwd(), "app/test_images/dataset-centered-image.png"))
    step += 1

    return positioned, centered

grey_bg = None
white_bg = None
noise_bg = None
random_color_bg = None
image = None

async def create_set(image_data: str, masks: List[MaskData], collection: str, token: str, caption: str, object_caption:str, bbox: tuple[int, int, int, int]):
    global image, grey_bg, white_bg, noise_bg, random_color_bg
    print(f"bbox: {bbox}")

    set_path = os.path.join(os.getcwd(), f"../models/user/{collection}/datasets/{token}")

    os.makedirs(set_path, exist_ok=True)

    filenames = {}

    def save_to_dataset(image, caption):
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(set_path, filename)
        print(f"path: {path}")
        image.save(path)
        filenames[filename] = caption

    step = 1
    image_data = extract_base64_data(image_data)
    bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(bytes)).convert("RGBA")
    print(f"image size: {image.size}")
    image.save(os.path.join(os.getcwd(), f"app/test_images/original-undedited-image.png"))

    composite_mask = Image.new(mode="L", size=image.size, color=0)
    composite_bbox: Tuple[int, int, int, int] = (image.width, image.height, 0, 0)

    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    for data in masks:

       mask = data["mask"] if data["include"] == True else data["inverted_mask"] if data["exclude"] == True else None

       if mask is None:
           continue

       b64 = extract_base64_data(mask)
       image_bytes = base64.b64decode(b64)

       mask = Image.open(BytesIO(image_bytes)).convert("RGBA")
       print(f"mask size: {mask.size}")
       mask.save(os.path.join(os.getcwd(), f"app/test_images/mask-data-mask-{step}.png"))
       step += 1

       alpha_channel = mask.getchannel("A").convert("L")
       mask = mask.convert("L")

       binary_mask = alpha_channel.point(lambda p: 255 if p > 0 else 0).convert("L")

       inverted_binary = ImageOps.invert(binary_mask)

       composite_mask.paste(binary_mask if data["include"] else inverted_binary, (data["bbox"][0], data["bbox"][1]), binary_mask)

       if data["include"]:
        bx1, by1, bx2, by2 = bbox
        dx1, dy1, dx2, dy2 = data["bbox"]
        cx1, cy1, cx2, cy2 = composite_bbox

        x1 = dx1 if dx1 < cx1 else cx1
        y1 = dy1 if dy1 < cy1 else cy1
        x2 = dx2 if dx2 > cx2 else cx2
        y2 = dy2 if dy2 > cy2 else cy2

        composite_bbox = tuple[int, int, int, int]([x1, y1, x2, y2])

    print(f"composite_bbox: {composite_bbox}")
    positioned = Image.new("L", size = image.size, color=0)
    print(f"positioned: {positioned}")
    print(f"composite_mask: {composite_mask}")
    positioned.paste(composite_mask, mask=composite_mask)
    positioned.save(os.path.join(os.getcwd(), "app/test_images/dataset-positioned-mask.png"))
    composite_mask = positioned

    # x1 = bx1
    # y1 = by1

    # composite_mask = ImageChops.composite(positioned, composite_mask, positioned)
    # composite_mask.save(os.path.join(os.getcwd(), f"app/test_images/dataset_composite_mask_{step}.png"))

    width_scale = 1024 / image.width
    height_scale = 1024 / image.height
    scale = min(width_scale, height_scale)
    resized_image = image.resize((int(image.width * scale), int(image.height * scale)))
    width_scale = 1024 / composite_mask.width
    height_scale = 1024 / composite_mask.height
    scale = min(width_scale, height_scale)
    composite_mask = composite_mask.resize((int(math.floor(composite_mask.width * scale)), int(math.floor(composite_mask.height * scale))))

    composite_bbox = [int(n * scale) for n in composite_bbox]


    padded_image = ImageOps.pad(resized_image, (1024, 1024), color=(0, 0, 0, 255))
    padded_image.save(os.path.join(os.getcwd(), "app/test_images/dataset-padded-image.png"))
    padded_mask = ImageOps.pad(composite_mask, (1024, 1024), color=0)

    x1, y1, x2, y2 = composite_bbox
    x1 = padded_mask.width // 2 - composite_mask.width // 2 + x1
    y1 = padded_mask.height // 2 - composite_mask.height // 2 + y1
    x2 = padded_mask.width // 2 - composite_mask.width // 2 + x2
    y2 = padded_mask.height // 2 - composite_mask.height // 2 + y2

    composite_bbox = [x1, y1, x2, y2]
    composite_mask = padded_mask
    image = padded_image

    # scale_x = 1024 / (x2 - x1)
    # scale_y = 1024 / (y2 - y1)
    # scale = min(scale_x, scale_y)

    # composite_bbox = [
    #     padded_image.width // 2 - resized_image.width // 2 + x1,
    #     padded_image.height // 2 - resized_image.height // 2 + y1,
    #     padded_image.width // 2 - resized_image.width // 2 + x2,
    #     padded_image.height // 2 - resized_image.height // 2 + y2
    # ]

    print(f"composite_bbox: {composite_bbox}")
    composite_mask = composite_mask.convert("L")
    composite_mask.save(os.path.join(os.getcwd(), f"app/test_images/dataset-padded-mask.png"))

    mid_x = image.width // 2
    mid_y = image.height // 2
    x1, y1, x2, y2 = composite_bbox
    mirrored_bbox = (mid_x - (x2 - mid_x), y1, mid_x - (x1 - mid_x), y2)
    flipped_bbox = (x1, mid_y - (y2 - mid_y), x2, mid_y - (y1 - mid_y))

    grey_bg = Image.new(mode="RGB", size=(1024, 1024), color=(127, 127, 127))

    def grey(image):
        positioned, centered = generate_images(image, grey_bg, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def grey_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(image), grey_bg, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def grey_flip(image):
        positioned, centered = generate_images(ImageOps.flip(image), grey_bg, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)


    white_bg = Image.new(mode="RGB", size=(1024, 1024), color=(255, 255, 255))

    def white(image):
        positioned, centered = generate_images(image, white_bg, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def white_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(image), white_bg, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def white_flip(image):
        positioned, centered = generate_images(ImageOps.flip(image), white_bg, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    # noise_bg = Image.effect_noise(size, 1.2).convert("RGB")

    rng = np.random.default_rng()

    noise_bg = Image.fromarray((rng.standard_normal(size=(1024, 1024, 3)) + 1 * 255).astype(np.uint8))

    def noise(image):

        print(f"noise_bg: {noise_bg}")
        noise_bg.save(os.path.join(os.getcwd(), "app/test_images/noise_bg.png"))

        positioned, centered = generate_images(image, noise_bg, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def noise_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(image), noise_bg, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def noise_flip(image):
        positioned, centered = generate_images(ImageOps.flip(image), noise_bg, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    random_color_bg = Image.new("RGB", size=(1024, 1024), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    random_color_bg = Image.new("RGB", size=(1024, 1024), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    def random_color(image):

        positioned, centered = generate_images(image, random_color_bg, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def random_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(image), random_color_bg, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def random_flip(image):
        positioned, centered = generate_images(ImageOps.flip(image), random_color_bg, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)


    enhancer = ImageEnhance.Contrast(image)
    contrast_variant = enhancer.enhance(random.random() / 2)
    grey_bg_contrast_variant_img = grey_bg.copy()

    def contrast(image):

        positioned, centered = generate_images(contrast_variant, grey_bg_contrast_variant_img, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def contrast_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(contrast_variant), grey_bg_contrast_variant_img, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def contrast_flip(image):
        positioned, centered = generate_images(ImageOps.flip(contrast_variant), grey_bg_contrast_variant_img, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)


    enhancer = ImageEnhance.Brightness(image)
    brightness_variant = enhancer.enhance(random.random() / 2)

    grey_bg_brightness_variant_img = grey_bg.copy()

    def brightness(image):


        positioned, centered = generate_images(brightness_variant, grey_bg_brightness_variant_img, composite_bbox, composite_mask)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def brightness_mirror(image):
        positioned, centered = generate_images(ImageOps.mirror(brightness_variant), grey_bg_brightness_variant_img, mirrored_bbox, ImageOps.mirror(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def brightness_flip(image):
        positioned, centered = generate_images(ImageOps.flip(brightness_variant), grey_bg_brightness_variant_img, flipped_bbox, ImageOps.flip(composite_mask))

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(positioned, object_caption)
        else:
            save_to_dataset(centered, object_caption)

    def full(image: Image.Image):
        global grey_bg
        full_image = grey_bg.copy()
        x1 = full_image.width // 2 - image.width // 2
        y1 = full_image.height // 2 - image.height // 2
        x2 = x1 + image.width
        y2 = y1 + image.height
        full_image.paste(image, (x1, y1, x2, y2))
        image = resize_image(full_image)

        n = random.randint(0,1)
        if n == 0:
            save_to_dataset(full_image, caption)
        else:
            save_to_dataset(ImageOps.mirror(image), caption)

    variants = [grey, grey_mirror, grey_flip, white, white_mirror, white_flip, noise, noise_mirror, noise_flip, random_color, random_mirror, random_flip, contrast, contrast_mirror, contrast_flip, brightness, brightness_mirror, brightness_flip ]

    random.choice(variants)(image)
    full(image)


    json_path = os.path.join(os.getcwd(), f"../models/user/{collection}/captions.json")

    if not os.path.exists(json_path):
        with open(json_path, mode="w") as f:
            f.write(json.dumps(filenames))
    else:
        with open(json_path, mode="r+") as f:
            d = json.loads(f.read())
            f.seek(0)
            for k, v in filenames.items():
                d[k] = v
            f.write(json.dumps(d))

    return {"status": "image added to dataset"}


class CustomPromptDataset(TextualInversionDataset):
    captions = {}

    def __init__(self, *args, captions: Dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.captions = captions

    def __get_item__(self, i):
        # example = super().__get_item__(i)

        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_holder = self.placeholder_token

        text = self.captions[self.image_paths[i % self.num_images].name]

        print(f"text: {text}")

        example["original_size"] = (image.height, image.width)

        image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
            image = transforms.functional.crop(image, y1, x1, h, w)

        example["crop_top_left"] = (y1, x1)

        example["input_ids_1"] = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example



class OneImageDataset(Dataset):

    def __init__(self, placeholder_token: str, initializer_token: str, collection: str, repeats: int, tokenizer_1, tokenizer_2, size=1024):

        self._mirror = False

        self._data_path = os.path.join(os.getcwd(), f'../models/user/{collection}/datasets/{placeholder_token}')
        self._filenames = os.listdir(self._data_path)


        self._placeholder_token = placeholder_token
        self._initializer_token = initializer_token
        self._repeats = repeats
        self._size = size
        self._tokenizer_1 = tokenizer_1
        self._tokenizer_2 = tokenizer_2
        self._jitter = T.ColorJitter(0.08, 0.08, 0.08, 0.02)
        self._collection = collection

        with open(os.path.join(self._data_path, '../../captions.json')) as f:
            self._captions = json.loads(f.read())

    def __len__(self):
        return self._repeats

    def __getitem__(self, i: int):
        transform_functions = [self.blur, self.mirror, self.jitter, self.resize]
        transform_functions = [t for t in [transform_functions[i:j] for i in range(len(transform_functions) - 1) for j in range(i + 1, len(transform_functions) + 1)]]
        example = {}

        filename = random.choice(self._filenames)

        prompt = None

        while prompt is None:
            try:
                prompt = self._captions[filename]['caption']

            except:
                prompt = None
                filename = random.choice(self._filenames)
                continue

        print(f"training prompt: {prompt}")

        image = Image.open(os.path.join(self._data_path, filename)).convert("RGB")

        bbox = self._captions[filename]['bbox']
        x1, y1, x2, y2 = bbox

        if self._mirror:
            bbox = (image.width - x2, y1, image.width - x1, y2)

        choices = random.choice(transform_functions)
        for t in choices:
            image = t(image)

        image = self.crop(image, bbox)

        example["original_size"] = (image.height, image.width)
        example["crop_top_left"] = (0, 0)
        example["input_ids_1"] = self._tokenizer_1(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_2"] = self._tokenizer_2(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image.save(os.path.join(os.getcwd(), f"app/training_images/{self._collection}-{datetime.now(timezone.utc)}.png"))

        image = np.asarray(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        print(f"example[\"pixel_values\"].size(): {example['pixel_values'].size()}")

        return example

    def blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.randint(1, 5)))

    def flip(self, image):
        return ImageOps.flip(image)

    def mirror(self, image):
        self._mirror = True
        return ImageOps.mirror(image)

    def resize(self, image):
        scale_x = self._size / image.width
        scale_y = self._size / image.height
        scale = max(scale_x, scale_y)
        return image.resize((int(math.ceil(image.width * scale)), int(math.ceil(image.height * scale))))

    def crop(self, image, bbox):
        width, height = image.size

        scale_x = self._size / width
        scale_y = self._size / height
        scale = max(scale_x, scale_y)

        image = image.resize((int(math.ceil(width * scale)), int(math.ceil(height * scale))))
        bbox = [int(n * scale) for n in bbox]

        x1, y1, x2, y2 = bbox
        x1 = x1 if x1 >= 0 else 0
        y1 = y1 if y1 >= 0 else 0
        x2 = x2 if x2 <= self._size else self._size
        y2 = y2 if y2 <= self._size else self._size
        w, h = x2 - x1, y2 - y1

        min_x = int(x1 - w * 0.25)
        min_x = min(min_x, image.width - self._size)
        min_x = min_x if min_x >= 0 else 0

        max_x = int(x1 + w * 0.25)
        max_x = max_x if max_x < image.width - self._size and max_x >= min_x else image.width - self._size

        print(f"min_x: {min_x} max_x: {max_x}")

        min_y = int(y1 - h * 0.25)
        min_y = min(min_y, image.height - self._size)
        min_y = min_y if min_y >= 0 else 0

        max_y = int(y1 + h * 0.25)
        max_y = max_y if max_y < image.height - self._size and max_y >= min_y else image.height - self._size

        print(f"min_y: {min_y} max_y: {max_y}")

        x1 = random.randint(min_x, max_x)
        y1 = random.randint(min_y, max_y)
        x2 = x1 + self._size
        y2 = y1 + self._size
        bbox = (x1, y1, x2, y2)
        print(f"bbox: {bbox}")
        w, h = x2 - x1, y2 - y1
        print(f"bbox size: {w}x{h}")
        image =  image.crop((x1, y1, x2, y2))
        print(f"image: {image}")
        self._mirror = False
        return image

    def jitter(self, image):
        return self._jitter(image)


def save_progress(tokenizer, tokenizer_2, text_encoder, text_encoder_2, token: str, weight_name: str, output_dir: str, step: int, optimizer: Optimizer, lr_scheduler: LRScheduler):
    os.makedirs(output_dir, exist_ok=True)
    print("saving checkpoint...")

    tok_id1 = tokenizer.convert_tokens_to_ids(token)
    tok_id2 = tokenizer_2.convert_tokens_to_ids(token)

    torch.save({
        "step": step,
        "token": token,
        "emb1": text_encoder.get_input_embeddings().weight[tok_id1],
        "emb2": text_encoder_2.get_input_embeddings().weight[tok_id2],
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
    }, os.path.join(output_dir, f"{weight_name}-{token}-step-{step}.pt"))


    emb1 = text_encoder.get_input_embeddings().weight[tok_id1].detach().cpu().unsqueeze(0)
    name1 = f"{weight_name}.safetensors"

    f = {}
    save_path_1 = os.path.join(output_dir, name1)
    if os.path.exists(save_path_1):
        f = load_file(save_path_1)

    f[token] = emb1
    save_file(f, save_path_1, {'format': 'pt'})

    f = {}


    emb2 = text_encoder_2.get_input_embeddings().weight[tok_id2].detach().cpu().unsqueeze(0)
    name2 = f"{weight_name}_2.safetensors"
    save_path_2 = os.path.join(output_dir, name2)
    if os.path.exists(save_path_2):
        f = load_file(save_path_2)
    f[token] = emb2
    save_file(f, save_path_2, {'format': 'pt'})


async def train(
    collection:str,
    token: str,
    initializer_token: str = "photo",
    resolution=1024,
    max_train_steps=100,
    num_training_steps=100,
    repeats=100,
    train_batch_size=1,
    gradient_accumulation_steps=1,
    lr=1e-4,
    lr_num_cycles=1,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
    num_epochs=1,
    lr_warmup_steps=20,
    validation_steps=100,
    batch_size=1,
    num_workers=0,
    resume_from_checkpoint="latest"
):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    num_processes = 1
    accelerator = None

    refiner = await get_refiner()

    if refiner is not None:
        if isinstance(refiner, DiffusionPipeline):
            refiner.to(torch.device("cpu"))
            refiner = None

    inpainter = await get_inpainter()

    if inpainter is not None:
        if isinstance(refiner, DiffusionPipeline):
            inpainter.to(torch.device("cpu"))
            inpainter = None

    controlnet = await get_controlnet()

    if controlnet is not None:
        if isinstance(controlnet, DiffusionPipeline):
            controlnet.to(torch.device("cpu"))
            controlnet = None

    pipe = await get_pipe()

    if pipe is not None:
        pipe.to("cpu")
        pipe = None

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

        accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="fp16",
            project_config=accelerator_project_config,
        )


    model_path = os.path.join(os.getcwd(), "../models/sdxl-base-1.0")
    output_dir = os.path.join(os.getcwd(), f"../models/user/{collection}")

    tokenizer_1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2=CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True)
    unet.eval()
    unet.enable_gradient_checkpointing()
    unet.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    vae.eval()
    vae.enable_tiling()
    vae.enable_slicing()
    vae.requires_grad_(False)

    if torch.cuda.is_available():
        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

    placeholder_tokens = [token]
    tokenizer_1.add_tokens(placeholder_tokens)
    tokenizer_2.add_tokens(placeholder_tokens)

    token_ids = tokenizer_1.encode(initializer_token, add_special_tokens=False)
    token_ids_2 = tokenizer_2.encode(initializer_token, add_special_tokens=False)

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(placeholder_tokens)
    initializer_token_id_2 = token_ids_2[0]
    placeholder_token_ids_2 = tokenizer_2.convert_tokens_to_ids(placeholder_tokens)

    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))

    token_embeds: torch.Tensor = cast(torch.Tensor, text_encoder_1.get_input_embeddings().weight.data)
    token_embeds_2: torch.Tensor = cast(torch.Tensor, text_encoder_2.get_input_embeddings().weight.data)

    with torch.no_grad():

        for token_id in placeholder_token_ids:
             token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        for token_id in placeholder_token_ids_2:
             token_embeds_2[token_id] = token_embeds_2[initializer_token_id_2].clone()


    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder_2.text_model.encoder.requires_grad_(False)
    text_encoder_2.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_2.text_model.embeddings.position_embedding.requires_grad_(False)

    text_encoder_1.gradient_checkpointing_enable()
    text_encoder_2.gradient_checkpointing_enable()

    if is_xformers_available():
        import xformers

        xformers_version = xformers.__version__
        if xformers_version == "0.0.16":
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()

    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        list(text_encoder_1.parameters()) + list(text_encoder_2.parameters()), lr=lr
    )

    placeholder_token = " ".join(tokenizer_1.convert_ids_to_tokens(placeholder_token_ids))
    print(f"tokenizer_1.get_added_vocab(): {tokenizer_1.get_added_vocab()}")
    print(f"placeholder_token: {placeholder_token}")

    train_dataset = OneImageDataset(
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
        collection=collection,
        repeats=repeats,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    weight_dtype = torch.float32
    if torch.cuda.is_available():

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    checkpoint = None
    global_step = 0
    first_epoch = 0
    path = None

    dirs = os.listdir(output_dir)
    checkpoints = [d for d in filter(lambda f: re.match(f"{collection}-{token}-step-\\d+?\\.pt", f), dirs)]
    if len(checkpoints) >= 1:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[3].split(".")[0]))

        path = checkpoints[-1]

    if path is None:
        print(
            f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        resume_from_checkpoint = None
        initial_global_step = 0
    else:
        print(f"Resuming from checkpoint {path}")

        if torch.cuda.is_available():
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[3].split(".")[0])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            checkpoint = torch.load(os.path.join(output_dir, path))
            token_embeds.to(device)
            token_embeds_2.to(device)
            checkpoint['emb1'].to(device, dtype=torch.float32)
            checkpoint['emb2'].to(device, dtype=torch.float32)
            token_embeds[token_ids] = checkpoint['emb1']
            token_embeds_2[token_ids_2] = checkpoint['emb2']

            global_step = checkpoint.get("step", 0)
            initial_global_step = global_step
            max_train_steps = max_train_steps + global_step

            optimizer = torch.optim.AdamW(list(text_encoder_1.parameters()) + list(text_encoder_2.parameters()), lr=lr)

            optimizer.load_state_dict(checkpoint["optimizer"])

            first_epoch = global_step // num_update_steps_per_epoch

    num_warmup_steps_for_scheduler = lr_warmup_steps * accelerator.num_processes if torch.cuda.is_available() else 1
    num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes if torch.cuda.is_available() else num_processes

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    if checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    text_encoder_1.train()
    text_encoder_2.train()

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * (accelerator.num_processes if torch.cuda.is_available() else 1 )* gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")


    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps"
    )

    if torch.cuda.is_available():
        orig_embeds_params = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()

        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

        orig_embeds_params_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()

        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")


    else:
        orig_embeds_params = text_encoder_1.get_input_embeddings().weight.data.clone()
        orig_embeds_params_2 = text_encoder_2.get_input_embeddings().weight.data.clone()

    text_encoder_1.get_input_embeddings().weight.requires_grad = True
    text_encoder_2.get_input_embeddings().weight.requires_grad = True

    print(f"dir(text_encoder_1.text_model.config): {dir(text_encoder_1.text_model.config)}")

    if torch.cuda.is_available():
        print(f"accelerator.device: {accelerator.device}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    to_pil = T.ToPILImage()

    save_steps = min(max_train_steps // 2, 50)
    print(f"text_encoder_1.get_added_vocab(): {tokenizer_1.get_added_vocab()}")

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()

        if torch.cuda.is_available():
            print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
            print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        try:
            for step, batch in enumerate(train_dataloader):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()


                images = batch["pixel_values"]
                images = images.squeeze(0)
                print(f"images.size(): {images.size()}")
                # images = torch.Tensor.numpy(images)
                print(f"images.shape: {images.shape}")
                images = to_pil(images)
                images = preprocess(images)

                if torch.cuda.is_available():
                    with accelerator.accumulate([text_encoder_1, text_encoder_2]):
                        # Convert images to latent space
                        images = images.unsqueeze(0)
                        latents = vae.encode(images.to(dtype=torch.float16)).latent_dist.sample()

                        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
                        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

                        latents = latents * vae.config.scaling_factor

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states_1 = (
                            text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                            .hidden_states[-2]
                        )

                        encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]

                        print(f"Line: {inspect.currentframe().f_lineno} Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
                        print(f"Line: {inspect.currentframe().f_lineno} Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

                        original_size = [
                            (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                            for i in range(train_batch_size)
                        ]
                        crop_top_left = [
                            (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                            for i in range(train_batch_size)
                        ]
                        target_size = (resolution, resolution)
                        add_time_ids = torch.cat(
                            [
                                torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                                for i in range(train_batch_size)
                            ]
                        )

                        added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                        encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)

                        # Predict the noise residual
                        model_pred = unet(
                            noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                        ).sample

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        accelerator.backward(loss)

                        for model_name, model in [("text_encoder_1", text_encoder_1),
                                                ("text_encoder_2", text_encoder_2),
                                                ("unet", unet)]:
                            for name, param in model.named_parameters():
                                if param.grad is not None and torch.isnan(param.grad).any():
                                    print(f"NaN in gradients of {name}")

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # Let's make sure we don't update any embedding weights besides the newly added token
                        index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
                        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                        index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool)
                        index_no_updates_2[min(placeholder_token_ids_2) : max(placeholder_token_ids_2) + 1] = False

                        with torch.no_grad():
                            accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[index_no_updates] = (
                                orig_embeds_params[index_no_updates]
                            )
                            accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[index_no_updates_2] = (
                                orig_embeds_params_2[index_no_updates_2]
                            )
                        if accelerator.sync_gradients:
                            images = []
                            progress_bar.update(1)
                            global_step += 1
                            if global_step % save_steps == 0:
                                weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                                save_path = os.path.join(output_dir, weight_name)
                                save_progress(tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, token=token, weight_name=collection, output_dir=output_dir, step=global_step, optimizer=optimizer, lr_scheduler=lr_scheduler)
                                weight_name = f"learned_embeds_2-steps-{global_step}.safetensors"
                                save_path = os.path.join(output_dir, weight_name)

                        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)

                        if global_step >= max_train_steps:
                            break
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                else:
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states_1 = (
                        text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                        .hidden_states[-2]
                    )

                    encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)

                    encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]

                    print(f"batch[\"original_size\"][0]: {batch["original_size"][0]}")
                    print(f"batch[\"original_size\"][1]: {batch["original_size"][1]}")
                    original_size = [
                        (batch["original_size"][i].item(), batch["original_size"][i].item())
                        for i in range(train_batch_size)
                    ]
                    print(f"original_size[0]: {original_size[0]} type(original_size[0]: {type(original_size[0])}")
                    crop_top_left = [
                        (batch["crop_top_left"][i].item(), batch["crop_top_left"][i].item())
                        for i in range(train_batch_size)
                    ]
                    target_size = (resolution, resolution)
                    print(f"target_size: {target_size}")
                    print(f"type(target_size):{type(target_size)}")
                    print(f"target_size[0]: {target_size[0]} target_size[1]: {target_size[1]}")
                    add_time_ids = torch.cat(
                        [
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(train_batch_size)
                        ]
                    )

                    added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                    encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                    print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")
                    print(f"noisy_latents.shape: {noisy_latents.shape}")
                    print(f"timesteps: {timesteps}")
                    # Predict the noise residual
                    # print(f"unet.weight.shape: {unet.weight.shape}")
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    print(f"loss: {loss}")

                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
                    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                    index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool)
                    index_no_updates_2[min(placeholder_token_ids_2) : max(placeholder_token_ids_2) + 1] = False

                    with torch.no_grad():

                        text_encoder_1.get_input_embeddings().weight[index_no_updates] = (
                            orig_embeds_params[index_no_updates]
                        )
                        text_encoder_2.get_input_embeddings().weight[index_no_updates_2] = (
                            orig_embeds_params_2[index_no_updates_2]
                        )

                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % save_steps == 0:

                    save_progress(tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, token=token, weight_name=collection, output_dir=output_dir, step=global_step, optimizer=optimizer, lr_scheduler=lr_scheduler)


                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                    print(f"Global step: {global_step}, loss: {loss.detach().item():.4f}")

                    if global_step >= max_train_steps:
                        break

        except Exception as e:

            save_progress(tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, token=token, weight_name=collection, output_dir=output_dir, step=global_step, optimizer=optimizer, lr_scheduler=lr_scheduler)

            raise e


        save_progress(tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, token=token, weight_name=collection, output_dir=output_dir, step=global_step, optimizer=optimizer, lr_scheduler=lr_scheduler)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status":"OK"}
