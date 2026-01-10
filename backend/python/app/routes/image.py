from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from app.services.db import UPLOADS_DIR, get_connection
from PIL import ImageOps, Image
from pydantic import BaseModel
import base64
from io import BytesIO
from cv2 import Canny
import numpy as np
import cv2

class PosterizeRequest(BaseModel):
    image_data: str
    bits: int
    

class EdgeRequest(BaseModel):
    image_data: str
    threshold1: float
    threshold2: float
    aperture_size: int
    l2_gradient: bool

router = APIRouter()

# UPLOADS_DIR = os.path.join(os.getcwd(), "app/uploads")

@router.get("/image/{filename}")
async def get_uploaded_image(filename: str):
    image_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    else:
        return FileResponse(image_path, media_type="image/jpeg")

@router.delete("/image/{filename}")
async def delete_uploaded_image(filename: str):
    image_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    else:
        os.remove(image_path)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM metadata WHERE filename = ?", (filename,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Image deleted successfully"}

@router.post('/image/posterize')
async def posterize(request: PosterizeRequest):
    image_data = request.image_data
    bits = request.bits
    
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes).convert('RGB')
    
    image = ImageOps.posterize(image, bits)
    
    buf = BytesIO()
    
    image.save(buf, format='PNG')
    
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode('utf-8')
    
    return {'image_data': image_data}


@router.post('/image/edges')
async def edges(request: EdgeRequest):
    print('api/image/edges')
    image_data, threshold1, threshold2, aperture_size, l2_gradient = request.image_data, request.threshold1, request.threshold2, request.aperture_size, request.l2_gradient
    
    # decode base64 image
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=int(aperture_size), L2gradient=l2_gradient)
    image = np.expand_dims(edges, axis=2).astype(np.uint8)
    image = np.concatenate([image, image, image], axis=2)

    image = Image.fromarray(image).convert('RGBA')
    
    buf = BytesIO()
    
    image.save(buf, format='PNG')
    
    buf.seek(0)
    
    image_data = base64.b64encode(buf.read()).decode('utf-8')
    
    return {'image_data': image_data}