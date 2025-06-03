import torch
import open_clip
from open_clip.transform import Compose
from PIL import Image, ImageFile
from torchvision import transforms
import os
from typing import cast


device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
)

preprocess_val = Compose(
    [
        # transforms.PILToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.2682954, 0.26130258, 0.27577711],
        ),
    ]
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

UPLOAD_DIR = "app/uploads"


def predict_clip_image(image: Image.Image, class_names: list[str]) -> dict:
    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        text_tokens = tokenizer(class_names).to(device)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    best_idx = cast(int, cast(torch.Tensor, similarity)[0].argmax().item())
    return {"predicted": class_names[best_idx], "scores": similarity[0].tolist()}
