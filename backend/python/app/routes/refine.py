from fastapi import APIRouter, WebSocket
from app.services.SDXL import refine, RefineRequest

router = APIRouter()

@router.post("/refine")
async def refine_img(request: RefineRequest):
    return await refine(request.prompt, request.image, request.strength, request.guidance_scale, request.negative_prompt, request.prompt_2, request.negative_prompt_2)

@router.websocket("/refine")
async def refine_img_ws(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()

        await ws.send_text(f"Message text was: {data}")
