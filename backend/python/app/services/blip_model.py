from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

processor = BlipProcessor.from_pretrained(f"{os.getcwd()}/../models/blip-processor", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(f"{os.getcwd()}/../models/blip-model")

def caption(image: Image.Image) -> dict:
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return {"caption": text}
