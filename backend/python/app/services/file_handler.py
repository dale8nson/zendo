from enum import unique
import os
import uuid
from fastapi import UploadFile
from starlette.exceptions import HTTPException

UPLOAD_DIR = "app/uploads"


async def save_upload_file(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if filename := file.filename:
        ext = os.path.splitext(filename)[-1]
        unique_name = f"{uuid.uuid4()}{ext}"
        destination = os.path.join(UPLOAD_DIR, unique_name)

        with open(destination, "wb") as out_file:
            content = await file.read()
            out_file.write(content)

        return unique_name
    raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")
