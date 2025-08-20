from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .routes import upload, metadata, predict, images, image, prompts, caption, score, masks, generate, refine, inpaint, ws, example
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from .services.db import init_db, UPLOADS_DIR
from contextlib import asynccontextmanager
from .services.clip_model import init_clip
from .services.SDXL import init_SDXL, init_refiner, init_inpainter, init_controlnet
from .services.SAM import init_SAM
from app.services.connection_manager import ConnectionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_clip()
    # await init_SDXL()
    # await init_refiner()
    # await init_inpainter()
    # await init_controlnet()
    # await init_upscaler()
    # await init_SAM()

    yield

app = FastAPI(lifespan=lifespan)

init_db()

app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

cwd = os.getcwd()

if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

else:
    print(f"[Warning] Uploads directory '{UPLOADS_DIR}' does not exist. Skipping upload mount.")

# app.include_router(example.router)

# app.include_router(predict.router, prefix="/api")
# app.include_router(images.router, prefix="/api")
# app.include_router(image.router, prefix="/api")
# app.include_router(score.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(refine.router, prefix="/api")
# app.include_router(masks.router, prefix="/api")
app.include_router(inpaint.router, prefix="/api")
# app.include_router(prompts.router, prefix="/api", tags=["prompts"])
app.include_router(ws.router, prefix="/ws")

manager = ConnectionManager()


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             await websocket.send_text(f"Message text was: {data}")
#     except WebSocketDisconnect:
#         print("Client disconnected")
