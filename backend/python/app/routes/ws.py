from fastapi import APIRouter, WebSocket
from app.services.SDXL import inpaint, InpaintRequest, refine, RefineRequest, generate, GenerateRequest
from app.services.connection_manager import ConnectionManager
from fastapi import WebSocketDisconnect
from typing import Dict, Any
import asyncio


manager = ConnectionManager()

class CallBack:
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def send(self, message) -> None:
        print(f"Sending message: {message}")
        await self.ws.send_json(message)

    def callback_fn(self, pipeline, step_index, timesteps) -> Dict[str, Any]:
        timestep_value = timesteps.item() if hasattr(timesteps, "item") else timesteps
        step_value = step_index.item() if hasattr(step_index, "item") else step_index
        print(f"Step {step_index}, Timestep {timesteps}")
        asyncio.create_task(self.send({"step": step_value, "timestep": timestep_value}))
        return {"step": step_value, "timestep": timestep_value}

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return self.callback_fn(pipeline, step_index, timestep)

router = APIRouter()

@router.websocket("/inpaint")
async def websocket_endpoint(websocket: WebSocket):
    callback_on_step_end = CallBack(ws)

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            response = await inpaint(message["image"], message["prompt"], message["mask"], message["strength"], message["guidance_scale"], message["negative_prompt"], message["prompt_2"], message["negative_prompt_2"], message["alpha"], message["noise"], message["noise_offset"], message["blur"], message["strict"], message["reverse_mask"], callback_on_step_end)
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e


@router.websocket("/masks")
async def websocket_endpoint(ws: WebSocket):
    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            response = await generate_masks(data_url=message["image"])
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        manager.disconnect(ws)
        raise e
