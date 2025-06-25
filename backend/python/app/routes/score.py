from fastapi import APIRouter
from app.services.clip_model import score_caption, ScoreRequest
from PIL import Image
from app.services.db import UPLOADS_DIR
import os

router = APIRouter()

@router.post("/score")
async def score(request: ScoreRequest) -> dict:
    if request.caption == "":
        return {score: 0}
    image = Image.open(os.path.join(UPLOADS_DIR, request.filename))
    result = await score_caption(image, request.caption)
    return  result
