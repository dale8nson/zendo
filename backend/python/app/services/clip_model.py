from sympy.core.basic import cacheit
import torch
import open_clip
from open_clip import CLIP, CoCa, CustomTextCLIP
from transformers import CLIPProcessor, CLIPModel
from open_clip.transform import Compose
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
from typing import cast, Any, Literal, Optional, List
import json
import numpy as np
from pydantic import BaseModel, ConfigDict
from fastapi import HTTPException
import math

BATCH_SIZE = 256

class TransformParams(BaseModel):
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    fit: Literal["cover", "contain"] = "contain"

class PredictRequest(BaseModel):
    id: int
    filename: str
    original_filename: str
    label: Optional[str] = None
    timestamp: str
    image_data: str
    width: int
    height: int
    transform: Optional[TransformParams] = None

embedded_text_features = None
global class_names, text_prompts


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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

model: Any = None
preprocess = None
tokenizer = open_clip.get_tokenizer("ViT-B-32")

async def get_clip_model():
    global model, tokenizer, preprocess
    if model is None:
        await init_clip()
    return model, processor, tokenizer

async def init_clip():
    global model, tokenizer, processor
    # model = CLIPModel.from_pretrained(os.path.join(os.getcwd(), '../models/clip/model'))
    # processor = CLIPProcessor.from_pretrained(os.path.join(os.getcwd(), '../models/clip/processor'))
    model, _, processor =  open_clip.create_model_and_transforms(
            model_name="ViT-bigG/14", pretrained="laion2b_s39b_b160k",
            cache_dir="../models/openclip"
        )

    model.to(device, dtype=torch.float32)
    model.eval()




async def init_model(text: List[str] = []) -> List[str]:
    global embedded_text_features, model

    user_model_path = os.path.join(os.getcwd(), '../models/user')

    model_dirs = [d for d in filter(lambda n: os.path.isdir(os.path.join(user_model_path, n)), os.listdir(user_model_path))]

    print(f'model_dirs: {model_dirs}')

    for model_dir in model_dirs:
        path = os.path.join(user_model_path, f"{model_dir}/clips.json")

        if os.path.exists(path):
            with open(path) as f:
                captions = json.loads(f.read())
                text.extend(captions)

    model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",

    )

    model.to(device, dtype=torch.float32)
    model.eval()

    if os.path.exists(f"{os.getcwd()}/../models/openclip/embeddings.npy"):
        print("Loading embeddings...")
        embedded_text_features = torch.from_numpy(np.load(f"{os.getcwd()}/../models/openclip/embeddings.npy")).to(device, dtype=torch.float32)
        print("Text embedding std deviation:", embedded_text_features.std().item())
        print("Number of prompts:", len(text))
        print("Shape of embeddings:", embedded_text_features.shape)

    else:
        print("Initializing model...")
        print("Loading vocabulary...")
        with torch.no_grad():
            print("Encoding text features...")
            text_features = []
            for i in range(0, len(text), BATCH_SIZE):
                print(f"Processing batch {(i // BATCH_SIZE) + 1} / {(len(text)//BATCH_SIZE) + 1}...")
                batch_prompts = text[i:i+BATCH_SIZE]
                batch_text_tokens = tokenizer(batch_prompts)
                batch_text_features = model.encode_text(batch_text_tokens)
                batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
                text_features.append(batch_text_features)

        print("Concatenating text features...")
        embedded_text_features = torch.cat(text_features, dim=0).to(device, dtype=torch.float32)
        print("Text embedding std deviation:", embedded_text_features.std().item())
        print("Number of prompts:", len(text))
        print("Shape of embeddings:", embedded_text_features.shape)
        print("Model initialized.")
        embeddings_path = os.path.join(os.getcwd(), "../models/openclip/embeddings.npy")
        embeddings_dir = os.path.dirname(embeddings_path)
        Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embedded_text_features.cpu().numpy())

    print(f"text: {text}")

    return text


async def predict_clip_image(image: Image.Image, text: List[str]) -> str:
    global embedded_text_features
    text = await init_model(text)
    print("Image size:", image.size)
    print("Image mode:", image.mode)
    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    print("Sum of image tensor:", img_tensor.sum().item())

    with torch.no_grad():
        print(f"img_tensor.dtype: {img_tensor.dtype}")
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print("image_features dtype:", image_features.dtype, "device:", image_features.device)
        print("text_features dtype:", embedded_text_features.dtype, "device:", embedded_text_features.device)
        similarity = (100.0 *  image_features @ embedded_text_features.T)
        # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    best_idx = cast(int, similarity[0].argmax().item())
    print(f"Predicted class: {text[best_idx]}")
    top_probs, top_idxs = similarity[0].topk(5)
    for i in range(5):
        print(f"{text[top_idxs[i]]}: {top_probs[i].item():.2f}")
    return text[best_idx]


async def score_caption(image: Image.Image, caption: str) -> float:
    global model, tokenizer

    if model is None:
        await init_clip()

    print(f"score_caption \"{caption}\" on device {device}")
    tokens = tokenizer([caption]).to(device)
    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        caption_features = model.encode_text(tokens)
        caption_features /= caption_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 *  image_features @ caption_features.T)
    print(f"Similarity: {similarity.item():.2f}")

    return similarity.item()


async def get_mask_label(image: str) -> dict:
    global model, tokenizer

    img_tensor = cast(torch.Tensor, preprocess_val(image))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)



async def predict_image(image: Image.Image ):

    scale_x = 224 / image.width
    scale_y = 224 / image.height
    scale = min(scale_x, scale_y)

    image = image.resize((int(math.ceil(image.width * scale)), int(math.ceil(image.height * scale))), resample=Image.Resampling.LANCZOS)




    result = await predict_clip_image(image)

    print(f"result: {result}")

    print(f"Prediction result: {result['predicted']}")

    return result
