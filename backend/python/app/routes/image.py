from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from email.policy import HTTP



router = APIRouter()

UPLOADS_DIR = os.path.join(os.getcwd(), "app/uploads")

@router.get("/image/{filename}")
async def get_uploaded_image(filename: str):
    image_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    else:
        return FileResponse(image_path, media_type="image/jpeg")
