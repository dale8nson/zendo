from fastapi import APIRouter, WebSocket
from app.services.SDXL import inpaint, InpaintRequest
from app.services.connection_manager import ConnectionManager
from fastapi import WebSocketDisconnect
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

@router.post("/inpaint")
async def inpaint_img(request: InpaintRequest):
    return await inpaint(request.image, request.prompt, request.mask, request.strength, request.guidance_scale, request.alpha)


manager = ConnectionManager()

class CallBack:
    def __init__(self, ws: WebSocket, loop):
        self.ws = ws
        self.loop = loop

    async def send(self, message) -> None:
        print(f"Sending message: {message}")
        await self.ws.send_json(message)
        await asyncio.sleep(0)

    def callback_fn(self, pipeline, step_index, timesteps) -> Dict[str, Any]:
        timestep_value = timesteps.item() if hasattr(timesteps, "item") else timesteps
        step_value = step_index.item() if hasattr(step_index, "item") else step_index
        print(f"Step {step_index}, Timestep {timesteps}")
        asyncio.run_coroutine_threadsafe(self.ws.send({"step": step_value, "timestep": timestep_value}), self.loop)
        return {}

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return self.callback_fn(pipeline, step_index, timestep)

@router.websocket("/inpaint")
async def inpaint_ws(ws: WebSocket):

    loop = asyncio.get_running_loop()
    callback_on_step_end = CallBack(ws, loop)

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            print(f"message keys: {message.keys()}")
            response = await inpaint(message["image"], message["prompt"], message["mask"], message["strength"], message["guidance_scale"], message["negative_prompt"], message["prompt_2"], message["negative_prompt_2"], message["alpha"], message["noise"], message["noise_offset"], message["blur"], message["strict"], message["reverse_mask"], callback_on_step_end)
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e
