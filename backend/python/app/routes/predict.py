from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import os
from app.services.clip_model import predict_clip_image
from app.services.metadata_handler import log_prediction_to_db
from PIL import Image

UPLOAD_DIR = "app/uploads"

router = APIRouter()


def crop_and_resize(
    image: Image.Image,
    x: float = 0.0,
    y: float = 0.0,
    scale: float = 1.0,
    fit: str = "contain",
    target_size: int = 224,
) -> Image.Image:
    w, h = image.size
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    image = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    offset_x = int(x * scaled_w)
    offset_y = int(y * scaled_h)
    image = image.crop(
        (offset_x, offset_y, offset_x + target_size, offset_y + target_size)
    )

    if fit == "contain":
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    elif fit == "cover":
        image = image.crop((0, 0, target_size, target_size))

    return image


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

@router.post("/predict")
async def predict_image(data: PredictRequest):
    image_path = os.path.join(UPLOAD_DIR, data.filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    image = Image.open(image_path)

    print("Transform provided:", data.transform)
    print("Fallback width:", data.width, "height:", data.height)

    # if data.transform:
    #     image = crop_and_resize(
    #         image,
    #         x=data.transform.x,
    #         y=data.transform.y,
    #         scale=data.transform.scale,
    #         fit=data.transform.fit,
    #     )
    # else:
    #     image = crop_and_resize(image,x=224 / 2 - data.width / 2, y=224 / 2 - data.height / 2, scale=1, fit="contain")

    result = await predict_clip_image(image)

    print(f"result: {result}")

    print(f"Prediction result: {result['predicted']}")

    log_prediction_to_db(
        data.filename, data.original_filename, result.get("predicted", ""), result.get("scores", []), data.width, data.height
    )

    return result
