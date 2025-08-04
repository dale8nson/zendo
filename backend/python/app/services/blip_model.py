from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageOps
import os
import base64
from io import BytesIO
from typing import Any, List
from pydantic.main import BaseModel


class CroppedImageCaptionRequest(BaseModel):
    image_data: str
    crop_box: List[int]


processor = BlipProcessor.from_pretrained(f"{os.getcwd()}/../models/blip-processor", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(f"{os.getcwd()}/../models/blip-model")

def caption(image: Image.Image) -> dict:
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return {"caption": text}

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
