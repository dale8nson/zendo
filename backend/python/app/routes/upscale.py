from fastapi import APIRouter
from app.services.upscale import composite_layers, upscale, UpscaleRequest

router = APIRouter()

@router.post("/upscale")
async def score(request: UpscaleRequest) -> dict:
    image_data, bbox = composite_layers(request.layers)

    result = upscale(image_data)
    return  {"bbox": bbox, "image_data": result }
