from fastapi import APIRouter
from app.services.upscale import composite_layers, upscale, UpscaleRequest

router = APIRouter()

@router.post("/upscale")
async def upscale_image(request: UpscaleRequest) -> dict:
    image_data = None
    if request.layers is not None:
        image_data, bbox = composite_layers(request.layers)
    else: image_data = request.image_data; bbox = request.bbox
    result = upscale(image_data)
    return  {"bbox": bbox, "image_data": result }
