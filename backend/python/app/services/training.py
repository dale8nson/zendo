import torch
import torch.nn.functional as F
import safetensors
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel, get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from pydantic.main import BaseModel
from typing import List, Tuple, Dict
from PIL import Image, ImageEnhance, ImageChops, ImageOps
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
from app.services.SDXL import pipe, refiner, inpainter
from app.services.textual_inversion.textual_inversion_sdxl import TextualInversionDataset

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
    masks: List[MaskData]
    collection: str
    token: str
    caption: str

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

    # scale_x = size / bg.width
    # scale_y = size / bg.height
    # scale = min(scale_x, scale_y)
    # bg = bg.resize((math.floor(bg.width * scale), math.floor(bg.height * scale)))
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

        placeholder_string = self.placeholder_token

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

def save_progress(text_encoder, placeholder_token_ids, placeholder_token, save_path, accelerator=None, safe_serialization=True):
    print("Saving embeddings")
    if accelerator is not None:
        learned_embeds = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        )
    else:
        learned_embeds = (
            text_encoder
            .get_input_embeddings()
            .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        )
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

async def train(
    collection:str,
    token: str,
    resolution=1024,
    max_train_steps=400,
    num_training_steps=400,
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
    save_steps=0,
    resume_from_checkpoint="latest"
):
    global refiner, inpainter
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MiB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MiB")
    print(f"accelerate available: {is_accelerate_available()}")

    x = torch.randn(1).cuda()
    print("Allocated:", torch.cuda.memory_allocated() / 1024 ** 2, "MiB")
    print("Reserved:", torch.cuda.memory_reserved() / 1024 ** 2, "MiB")

    if refiner is not None:
        refiner.to(torch.device("cpu"))
    if inpainter is not None:
        inpainter.to(torch.device("cpu"))

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = os.path.join(os.getcwd(), "../models/sdxl-base-1.0")
    output_dir = os.path.join(os.getcwd(), f"../models/user/{collection}")

    if torch.cuda.is_available():
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="fp16" if (torch.cuda.is_available()) else "no",
            project_config=accelerator_project_config,
        )

    tokenizer_1=CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "../models/sdxl-base-1.0"), subfolder="tokenizer")
    tokenizer_2=CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")


    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

    vae.use_slicing = True
    vae.use_tiling = True


    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, use_safetensors=True).to(device)

    initializer_token = re.search("(?<=<).+(?=>)", token)[0]
    placeholder_tokens = [token]
    tokenizer_1.add_tokens(placeholder_tokens)
    tokenizer_2.add_tokens(placeholder_tokens)
    token_ids = tokenizer_1.encode(initializer_token, add_special_tokens=False)
    token_ids_2 = tokenizer_2.encode(initializer_token, add_special_tokens=False)

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(placeholder_tokens)
    initializer_token_id_2 = token_ids_2[0]
    placeholder_token_ids_2 = tokenizer_2.convert_tokens_to_ids(placeholder_tokens)

    # old_weight_1 = text_encoder_1.get_input_embeddings().weight.detach().cpu().clone()
    # old_weight_2 = text_encoder_2.get_input_embeddings().weight.detach().cpu().clone()

    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))

    token_embeds = text_encoder_1.get_input_embeddings().weight.data
    token_embeds_2 = text_encoder_2.get_input_embeddings().weight.data

    with torch.no_grad():

         for token_id in placeholder_token_ids:
             token_embeds[token_id] = token_embeds[initializer_token_id].clone().to(device)
         for token_id in placeholder_token_ids_2:
             token_embeds_2[token_id] = token_embeds_2[initializer_token_id_2].clone().to(device)
        # Step 1: Resize (if not done already)
        # text_encoder_1.resize_token_embeddings(len(tokenizer_1))
        # text_encoder_2.resize_token_embeddings(len(tokenizer_2))
        # Step 2: Get original weights
        # old_weight_1 = text_encoder_1.get_input_embeddings().weight.detach().clone().to(device)
        # old_weight_2 = text_encoder_2.get_input_embeddings().weight.detach().clone().to(device)

        # Step 3: Create a fresh new Embedding layer on MPS
        # new_embedding_1 = torch.nn.Embedding(old_weight_1.size(0), old_weight_1.size(1)).to(device)
        # new_embedding_1.weight.data.copy_(old_weight_1)

        # new_embedding_2 = torch.nn.Embedding(old_weight_2.size(0), old_weight_2.size(1)).to(device)
        # new_embedding_2.weight.data.copy_(old_weight_2)

        # Step 4: Assign the new embedding
        # text_encoder_1.text_model.embeddings.token_embedding = new_embedding_1
        # text_encoder_2.text_model.embeddings.token_embedding = new_embedding_2

        # old_weight = text_encoder_1.text_model.embeddings.token_embedding.weight.detach().clone()
        # from torch import nn
        # new_embedding = nn.Embedding.from_pretrained(old_weight, freeze=False)
        # text_encoder_1.text_model.embeddings.token_embedding = new_embedding

        # weight_1 = text_encoder_1.get_input_embeddings().weight
        # weight_2 = text_encoder_2.get_input_embeddings().weight

        # print("Embedding weight device (1):", weight_1.device)
        # print("Embedding weight device (2):", weight_2.device)

        # weight_1[:old_weight_1.size(0)] = old_weight_1.to(device)
        # weight_2[:old_weight_2.size(0)] = old_weight_2.to(device)

        # for token_id in placeholder_token_ids:
        #     embedding = weight_1[initializer_token_id].detach().clone().to(device)
        #     weight_1[token_id].copy_(embedding)
        #     print(f"Token {token_id} device:", weight_1[token_id].device)

        # for token_id in placeholder_token_ids_2:
        #     embedding2 = weight_2[initializer_token_id_2].detach().clone().to(device)
        #     weight_2[token_id].copy_(embedding2)
        #     print(f"Token {token_id} device:", weight_2[token_id].device)




    vae.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder_2.text_model.encoder.requires_grad_(False)
    text_encoder_2.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_2.text_model.embeddings.position_embedding.requires_grad_(False)

    text_encoder_1.gradient_checkpointing_enable()
    text_encoder_2.gradient_checkpointing_enable()

    # text_encoder_1.to(device, dtype=weight_dtype)
    # text_encoder_2.to(device, dtype=weight_dtype)

    if is_xformers_available():
        import xformers

        xformers_version = xformers.__version__
        if xformers_version == "0.0.16":
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class([
        text_encoder_1.text_model.embeddings.token_embedding.weight,
        text_encoder_2.text_model.embeddings.token_embedding.weight,
    ],
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay
    )

    placeholder_token = " ".join(tokenizer_1.convert_ids_to_tokens(placeholder_token_ids))

    json_path = os.path.join(os.getcwd(), f"../models/user/{collection}/captions.json")

    captions = {}

    with open(json_path, mode="r") as f:
        captions = json.loads(f.read())

    train_dataset = CustomPromptDataset(
        captions=captions,
        data_root=os.path.join(os.getcwd(), f"../models/user/{collection}/datasets/{token}"),
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=1024,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token=placeholder_token,
        center_crop=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    num_warmup_steps_for_scheduler = lr_warmup_steps * accelerator.num_processes if torch.cuda.is_available() else 1

    num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes if torch.cuda.is_available() else num_processes

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps,
    )

    text_encoder_1.train()
    text_encoder_2.train()

    if torch.cuda.is_available():
        text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device if (torch.cuda.is_available()) else device, dtype=weight_dtype)
    vae.to(accelerator.device if (torch.cuda.is_available()) else device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device if (torch.cuda.is_available()) else device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * (accelerator.num_processes if torch.cuda.is_available() else 1 )* args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if resume_from_checkpoint != "latest":
        path = os.path.basename(resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            checkpoint = torch.load(os.path.join(output_dir, path),map_location=device)

            global_step = int(path.split("-")[1])

            text_encoder_1.load_state_dict(checkpoint["model"])
            optimizer = torch.optim.AdamW(text_encoder_1.parameters(), lr=lr).to(device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

            global_step = checkpoint.get("global_step", 0)
            initial_global_step = global_step

            first_epoch = global_step // num_update_steps_per_epoch


    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps"
    )

    if torch.cuda.is_available():
        orig_embeds_params = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()
        orig_embeds_params_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()
    else:
        orig_embeds_params = text_encoder_1.get_input_embeddings().weight.data.clone().to(device)
        orig_embeds_params_2 = text_encoder_2.get_input_embeddings().weight.data.clone().to(device)

    # with torch.no_grad():
    #     dummy_input = torch.tensor([placeholder_token_ids[0]], dtype=torch.long, device=device).to(device)
    #     dummy_mask = torch.tensor([[1]], dtype=torch.long, device=device).to(device)
    #     _ = text_encoder_1(input_ids=dummy_input, attention_mask=dummy_mask, output_hidden_states=True)

    # for param in text_encoder_1.parameters():
    #     param.requires_grad = False

    # for param in text_encoder_2.parameters():
    #     param.requires_grad = False

    text_encoder_1.get_input_embeddings().weight.requires_grad = True  # Just the embeddings
    text_encoder_2.get_input_embeddings().weight.requires_grad = True

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                with accelerator.accumulate([text_encoder_1, text_encoder_2]):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
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
                        .to(dtype=weight_dtype)
                    )
                    encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                    encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
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
                    ).to(accelerator.device, dtype=weight_dtype)
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
                            save_progress(
                                text_encoder_1,
                                placeholder_token_ids,
                                accelerator,
                                placeholder_token,
                                save_path,
                                safe_serialization=True,
                            )
                            weight_name = f"learned_embeds_2-steps-{global_step}.safetensors"
                            save_path = os.path.join(output_dir, weight_name)
                            save_progress(
                                text_encoder_2,
                                placeholder_token_ids_2,
                                accelerator,
                                placeholder_token,
                                save_path,
                                safe_serialization=True,
                            )
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= max_train_steps:
                        break
                accelerator.wait_for_everyone()
                accelerator.end_training()
            else:
                latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample().detach()
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
                    text_encoder_1(batch["input_ids_1"].to(device), output_hidden_states=True)
                    .hidden_states[-2]
                    .to(dtype=weight_dtype)
                )
                encoder_output_2 = text_encoder_2(batch["input_ids_2"].to(device), output_hidden_states=True)
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
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
                ).to(device, dtype=weight_dtype)
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

                accelerator.backward(loss) if torch.cuda.is_avalable() else loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool)
                index_no_updates_2[min(placeholder_token_ids_2) : max(placeholder_token_ids_2) + 1] = False

                with torch.no_grad():
                    if torch.cuda.is_available():
                        accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[index_no_updates] = (
                            orig_embeds_params[index_no_updates]
                        )
                        accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[index_no_updates_2] = (
                            orig_embeds_params_2[index_no_updates_2]
                        )
                    else:
                        text_encoder_1.get_input_embeddings().weight[index_no_updates] = (
                            orig_embeds_params[index_no_updates]
                        )
                        text_encoder_2.get_input_embeddings().weight[index_no_updates_2] = (
                            orig_embeds_params_2[index_no_updates_2]
                        )


                if torch.cuda.is_available():
                    if accelerator.sync_gradients:
                        images = []
                        progress_bar.update(1)
                        global_step += 1
                        if global_step % args.save_steps == 0:
                            weight_name = f"{token}.safetensors"
                            save_path = os.path.join(output_dir, weight_name)
                            save_progress(
                                text_encoder_1,
                                placeholder_token_ids,
                                placeholder_token,
                                save_path=save_path,
                                accelerator=accelerator,
                                safe_serialization=True,
                            )
                            weight_name = f"{token}_2.safetensors"
                            save_path = os.path.join(output_dir, weight_name)
                            save_progress(
                                text_encoder_2,
                                placeholder_token_ids_2,
                                placeholder_token,
                                save_path=save_path,
                                accelerator=accelerator,
                                safe_serialization=True,
                            )
                    accelerator.wait_for_everyone()

                    accelerator.end_training()

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                print(f"Global step: {global_step}, loss: {loss.detach().item():.4f}")

                if global_step >= max_train_steps:
                    break



        # pipe.load_textual_inversion(os.path.join(output_dir, token))
        if refiner is not None:
            refiner.to(device)

        if inpainter is not None:
            inpainter.to(device)
            inpainter.load_textual_inversion(os.path.join(output_dir, token))

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status":"OK"}

    # torch.save({
    #     "model": text_encoder_1.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": lr_scheduler.state_dict(),
    #     "global_step": global_step,
    # }, "path/to/checkpoint.pth")
