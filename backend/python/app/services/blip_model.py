from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor
from PIL import Image, ImageOps
import os
import base64
from io import BytesIO
from typing import Any, List, cast
from pydantic.main import BaseModel
from .clip_model import get_clip_model, preprocess_val, predict_clip_image
import json
import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms

class CroppedImageCaptionRequest(BaseModel):
    image_data: str
    crop_box: List[int]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

processor = BlipProcessor.from_pretrained(f"{os.getcwd()}/../models/blip-processor", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(f"{os.getcwd()}/../models/blip-model")

async def caption(image: Image.Image) -> dict:

    global processor, model
    # clip_model, clip_processor, clip_tokenizer = await get_clip_model()

    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.decode(outputs[0], skip_special_tokens=True)

    scale_x = 224 / image.width
    scale_y = 224 / image.height
    scale = max(scale_x, scale_y)

    image = image.resize((int(image.width * scale), int(image.height * scale)))
    image = ImageOps.pad(image, (224, 224), color=(170, 170, 170))
    print(f"type(text): {type(text)}")

    caption = await predict_clip_image(image=image, text=[text])


    return {"caption": caption}

def cropped_image_caption(image_data: str, crop_box: tuple[int, int, int, int]):
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image.save(f"{os.getcwd()}/app/test_images/blip-original_image.png")

    image.save(f"{os.getcwd()}/app/test_images/blip-cropped_image.png")
    image = image.crop(crop_box)
    image = ImageOps.pad(image, (1024, 1024), centering=(0.5, 0.5), color=0)
    image.save(f"{os.getcwd()}/app/test_images/blip-padded_image.png")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
