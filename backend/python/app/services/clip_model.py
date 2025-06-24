import torch
import open_clip
from open_clip.transform import Compose
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
from typing import cast
import json
import numpy as np
from pydantic import BaseModel, ConfigDict

BATCH_SIZE = 256

global embedded_text_features
global class_names, text_prompts

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

class ScoreRequest(BaseModel):
    filename: str
    caption: str

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

async def init_model():
    global class_names, text_prompts, embedded_text_features
    class_names = ["Korean woman wearing a white baseball cap, with a plush elephant toy attached at her waist, holding a gun", "Korean woman wearing a white baseball cap","asian", "asian woman", "black model walking on catwalk", "south asian man wearing a striped blue and green knitted cardiagn and orange trousers", "high fashion", "male asian high fashion model wearing an impractical grey suit"]
    text_prompts = [f"a photo of a {label}" for label in class_names]

    if os.path.exists(f"{os.getcwd()}/../models/openclip/embeddings.npy"):
        print("Loading embeddings...")
        embedded_text_features = torch.from_numpy(np.load(f"{os.getcwd()}/../models/openclip/embeddings.npy")).to(device)
        print("Text embedding std deviation:", embedded_text_features.std().item())
        print("Number of prompts:", len(text_prompts))
        print("Shape of embeddings:", embedded_text_features.shape)

    else:
        print("Initializing model...")
        print("Loading vocabulary...")
        with torch.no_grad():
            print("Encoding text features...")
            text_features = []
            for i in range(0, len(text_prompts), BATCH_SIZE):
                print(f"Processing batch {(i // BATCH_SIZE) + 1} / {(len(text_prompts)//BATCH_SIZE) + 1}...")
                batch_prompts = text_prompts[i:i+BATCH_SIZE]
                batch_text_tokens = tokenizer(batch_prompts)
                batch_text_features = model.encode_text(batch_text_tokens)
                batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
                text_features.append(batch_text_features)
        print("Concatenating text features...")
        embedded_text_features = torch.cat(text_features, dim=0).to(device)
        print("Text embedding std deviation:", embedded_text_features.std().item())
        print("Number of prompts:", len(text_prompts))
        print("Shape of embeddings:", embedded_text_features.shape)
        print("Model initialized.")
        embeddings_path = os.path.join(os.getcwd(), "../models/openclip/embeddings.npy")
        embeddings_dir = os.path.dirname(embeddings_path)
        Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embedded_text_features.cpu().numpy())

# with open(f"{os.getcwd()}/../models/openclip/vocab.json", "r") as f:
#     obj = json.load(f)
#     class_names = [key for (key, _) in obj.items()]
#     text_prompts = [f"a photo of a {label}" for label in class_names]

# with torch.no_grad():
#     text_tokens = tokenizer(text_prompts)
#     text_features = model.encode_text(text_tokens)
#     text_features /= text_features.norm(dim=-1, keepdim=True)


UPLOAD_DIR = "app/uploads"

async def predict_clip_image(image: Image.Image) -> dict:
    global embedded_text_features
    if embedded_text_features is None:
        raise RuntimeError("Model has not been initialised. Please run init_model() first.")
    print("Image size:", image.size)
    print("Image mode:", image.mode)
    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print("Sum of image tensor:", img_tensor.sum().item())

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print("image_features dtype:", image_features.dtype, "device:", image_features.device)
        print("text_features dtype:", embedded_text_features.dtype, "device:", embedded_text_features.device)
        similarity = (100.0 *  image_features @ embedded_text_features.T)
        # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    global text_prompts
    best_idx = cast(int, similarity[0].argmax().item())
    print(f"Predicted class: {text_prompts[best_idx]}")
    top_probs, top_idxs = similarity[0].topk(5)
    for i in range(5):
        print(f"{text_prompts[top_idxs[i]]}: {top_probs[i].item():.2f}")
    return {"predicted": text_prompts[best_idx], "scores": similarity[0].tolist()[0:4]}

async def score_caption(image: Image.Image, caption: str) -> dict:
    print("score_caption")
    tokens = tokenizer([caption])
    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        caption_features = model.encode_text(tokens)
        caption_features /= caption_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 *  image_features @ caption_features.T)
    print(f"Similarity: {similarity.item():.2f}")
    return {"score": similarity.item()}
