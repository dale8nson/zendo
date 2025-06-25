from fastapi import APIRouter
from app.services.SDXL import generate, GenerateRequest

router = APIRouter()

@router.post("/generate")
async def generate_image(request: GenerateRequest) -> dict:
    result = await generate(request.prompt)
    return result
