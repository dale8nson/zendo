from fastapi import APIRouter, HTTPException
from app.services.blip_model import caption
from app.routes.predict import PredictRequest, UPLOAD_DIR
import os
from PIL import Image

router = APIRouter()

@router.post("/caption")
async def create_caption(data: PredictRequest):

    image_path = os.path.join(UPLOAD_DIR, data.filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    image = Image.open(image_path)

    return caption(image)
