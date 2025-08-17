from fastapi import APIRouter
from app.services.SDXL import TokenizeRequest, tokenize

router = APIRouter()

@router.post("/tokenize")
async def score(request: TokenizeRequest) -> dict:

    result = await tokenize(request.text)
    return  result
