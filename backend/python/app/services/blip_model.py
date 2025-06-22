from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def caption(image: Image.Image) -> dict:
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return {"caption": text}
