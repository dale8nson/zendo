from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Annotated
from fastapi.responses import JSONResponse
import shutil
from app.services.metadata_handler import save_metadata_entry
import os
from datetime import datetime
import uuid
from app.services.db import UPLOADS_DIR
from PIL import Image


# from safetensors.torch import save_model

router = APIRouter()
# HERE = Path(__file__).resolve().parent
# UPLOADS_DIR = HERE.parent / "uploads"
# UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_image(file: Annotated[UploadFile, Form()], collection: Annotated[str, Form()]):
    print(f"{__file__}: UPLOADS_DIR: {UPLOADS_DIR}")
    original_filename = file.filename
    if original_filename:
        name, ext = os.path.splitext(original_filename)
        generated_id = uuid.uuid4().hex
        new_filename = f"{generated_id}{ext}"
        save_path = os.path.join(UPLOADS_DIR, new_filename)
        print(f"save_path: {save_path}")
    else: raise HTTPException(status_code=400, detail="No file provided")

    if not UPLOADS_DIR.exists():
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    try:

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with Image.open(file.file) as img:
            width, height = img.size
            print(f"img width:{width}, height:{height}")
        metadata = {
            "filename": new_filename,
            "timestamp": datetime.utcnow().isoformat(),
            "original_filename": original_filename,
            "width": width,
            "height": height,
            "collection": collection
        }

        print(metadata)
        await save_metadata_entry(metadata)

        return JSONResponse(content={"status": "success", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
