from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.SAM import generate_masks, MaskRequest
from app.services.connection_manager import ConnectionManager
from fastapi import WebSocketDisconnect, WebSocket


router = APIRouter()

@router.post("/masks")
async def get_masks(data: MaskRequest):
    masks = await generate_masks(data_url=data.image)
    return JSONResponse(masks)


manager = ConnectionManager()

# @router.websocket("/masks")
# async def websocket_endpoint(ws: WebSocket):
#     global manager
#     await manager.connect(ws)

#     try:
#         async for message in ws.iter_json():
#             await ws.send_json({"status": "started"})
#             response = await generate_masks(data_url=message["image"])
#             await ws.send_json(response)

#     except WebSocketDisconnect:
#         manager.disconnect(ws)
#     except Exception as e:
#         manager.disconnect(ws)
#         raise e
