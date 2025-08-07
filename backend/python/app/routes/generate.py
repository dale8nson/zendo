from fastapi import APIRouter
from app.services.SDXL import generate, GenerateRequest

router = APIRouter()

@router.post("/generate")
async def generate_image(request: GenerateRequest) -> dict:
    result = await generate(request.prompt, request.iterations, request.guidance_scale, request.negative_prompt, request.prompt_2, request.negative_prompt_2, request.ipAdapterImage)
    return result
