from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import shutil
from app.services.metadata_handler import save_metadata_entry
import os
from datetime import datetime
import uuid
from pathlib import Path
from app.services.db import UPLOADS_DIR


# from safetensors.torch import save_model

router = APIRouter()
# HERE = Path(__file__).resolve().parent
# UPLOADS_DIR = HERE.parent / "uploads"
# UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_image(file: UploadFile = File(...), label: str = Form(...)):
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
        metadata = {
            "filename": new_filename,
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
            "original_filename": original_filename
        }

        print(metadata)
        save_metadata_entry(metadata)

        return JSONResponse(content={"status": "success", "filename": file.filename, "label": label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
