from fastapi import APIRouter, HTTPException
from rembg import remove
import base64
from io import BytesIO

router = APIRouter()

def extract_base64_data(data_url: str) -> str:
    # Remove header if present
    if ',' in data_url:
        b64 = data_url.split(',', 1)[1]
    else:
        b64 = data_url
    b64 = b64.strip().replace('\n', '').replace(' ', '')
    # Fix base64 padding
    pad = len(b64) % 4
    if pad:
        b64 += '=' * (4 - pad)
    return b64


@router.post("/removebg")
async def remove_background(image: str):
    b64 = extract_base64_data(image)
    bytes = base64.b64decode(b64)
    bytes = remove(bytes)
    bytes = base64.b64encode(BytesIO(bytes))
