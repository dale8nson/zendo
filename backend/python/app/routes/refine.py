from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.SDXL import refine, RefineRequest
from app.services.connection_manager import ConnectionManager
import asyncio
from typing import Dict
from typing import Any

router = APIRouter()

# @router.post("/refine")
# async def refine_img(request: RefineRequest):
#     return await refine(request.prompt, request.image, request.strength, request.guidance_scale, request.negative_prompt, request.prompt_2, request.negative_prompt_2)

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

@router.websocket("/refine")
async def refine_ws(ws: WebSocket):

    loop = asyncio.get_running_loop()
    callback_on_step_end = CallBack(ws, loop)

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            print(f"message keys: {message.keys()}")
            response = await refine(message["prompt"], message["layers"], message["strength"], message["inference_steps"], message["guidance_scale"], message["negative_prompt"], message["prompt_2"], message["negative_prompt_2"], message["remove_background"], callback_on_step_end = callback_on_step_end)
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e
