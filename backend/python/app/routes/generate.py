from fastapi import APIRouter
from app.services.SDXL import generate, GenerateRequest

router = APIRouter()

@router.post("/generate")
async def generate_image(request: GenerateRequest) -> dict:
    print('generate_image')
    result = await generate(request.prompt, request.iterations, request.guidance_scale, request.negative_prompt, request.prompt_2, request.negative_prompt_2, request.ip_adapter_image, request.use_face_id, request.bbox, request.remove_background, request.use_ip_adapter_image, request.refiner_strength, request.seed)
    return result
