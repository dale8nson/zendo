from fastapi import APIRouter, WebSocket
from app.services.SDXL import inpaint, InpaintRequest, refine, RefineRequest, generate, GenerateRequest, pipe, refiner, inpainter
from app.services.SAM import generate_masks, generate_mask
from app.services.training import create_set, train
from app.services.connection_manager import ConnectionManager
from fastapi import WebSocketDisconnect
from typing import Dict, Any
import websockets
import asyncio
import os


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

# @router.websocket("/inpaint")
# async def inpaint_websocket(ws: WebSocket):
#     callback_on_step_end = CallBack(ws)

#     global manager
#     await manager.connect(ws)

#     try:
#         async for message in ws.iter_json():
#             await ws.send_json({"status": "started"})
#             response = await inpaint(message["image"], message["prompt"], message["mask"], message["strength"], message["guidance_scale"], message["negative_prompt"], message["prompt_2"], message["negative_prompt_2"], message["alpha"], message["noise"], message["noise_offset"], message["blur"], message["strict"], message["reverse_mask"], callback_on_step_end)
#             await ws.send_json(response)

#     except WebSocketDisconnect:
#         manager.disconnect(ws)
#     except Exception as e:
#         await ws.send_json({"error": str(e)})
#         raise e


@router.websocket("/mask")
async def mask_ws(ws: WebSocket):
    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            response = await generate_mask(image=message["image"], bbox=message["bbox"])

            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        manager.disconnect(ws)
        raise e

@router.websocket("/masks")
async def masks_ws(ws: WebSocket):
    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})

            response = await generate_masks(image_data=message["image"])

            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        manager.disconnect(ws)
        raise e


# @router.websocket("/img2img")
# async def img2img_ws(ws: WebSocket):

#     loop = asyncio.get_running_loop()
#     callback_on_step_end = CallBack(ws, loop)

#     global manager
#     await manager.connect(ws)

#     try:
#         async for message in ws.iter_json():
#             await ws.send_json({"status": "started"})

#             response = await generate_masks(image_data=message["image"])

#             await ws.send_json(response)

    # except WebSocketDisconnect:
    #     manager.disconnect(ws)
    # except Exception as e:
    #     await ws.send_json({"error": str(e)})
    #     manager.disconnect(ws)
    #     raise e


@router.websocket("/img2img")
async def img2img_ws(ws: WebSocket):

    loop = asyncio.get_running_loop()
    callback_on_step_end = CallBack(ws, loop)

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            print(f"message keys: {message.keys()}")
            response = await refine(message["prompt"], message["image"], message["strength"], message["inference_steps"], message["guidance_scale"], message["negative_prompt"], message["prompt_2"], message["negative_prompt_2"], message["refiner_prompt"], message["refiner_negative_prompt"], message["refiner_prompt_2"], message["refiner_negative_prompt_2"], callback_on_step_end = callback_on_step_end)
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e

@router.websocket("/dataset")
async def dataset_ws(ws: WebSocket):

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            print(f"message keys: {message.keys()}")
            response = await create_set(message["image_data"], message["masks"], message["collection"], message["token"], message["caption"], message["object_caption"], message["bbox"])
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e

@router.websocket("/train")
async def train_ws(ws: WebSocket):

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            await ws.send_json({"status": "started"})
            response = await train(message["collection"], message["token"])
            await ws.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e


async def websocket_send(uri: str, message: str):
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
        response = await websocket.recv()
        return response

# @router.websocket("/load_inversion")
# async def load_inversion(ws: WebSocket):
#     global manager

#     await manager.connect(ws)

#     try:
#         async for message in ws.iter_json():
#             collection = message["collection"]
#             token = message["token"]

#             # websocket_send(uri="ws:")

#             with open(path, "rb") as f:
#                 await ws.send_bytes(f.read())

#     except WebSocketDisconnect:
#         manager.disconnect(ws)
#     except Exception as e:
#         await ws.send_json({"error": str(e)})
#         raise e


@router.websocket("/inversion")
async def inversion_ws(ws: WebSocket):

    global manager
    await manager.connect(ws)

    try:
        async for message in ws.iter_json():
            collection = message["collection"]
            token = message["token"]

            path = os.path.join(os.getcwd(), f"../models/user/{collection}/{token}.safetensors")

            with open(path, "rb") as f:
                await ws.send_bytes(f.read())

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        await ws.send_json({"error": str(e)})
        raise e
