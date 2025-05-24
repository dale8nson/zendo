from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.file_handler import save_upload_file

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    filename = await save_upload_file(file)
    return {"filename": filename}
