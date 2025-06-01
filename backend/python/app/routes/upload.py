from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.file_handler import save_upload_file
from app.services.metadata_handler import save_metadata_entry
import sqlite3
from safetensors.torch import save_model

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    filename = await save_upload_file(file)

    save_metadata_entry({
        "filename": filename,
        "label": None
    })
    return {"filename": filename}
