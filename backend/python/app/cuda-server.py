from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .routes import upload, metadata, predict, caption, score, masks, ws
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager
from .services.clip_model import init_clip
from .services.SAM import init_SAM
from app.services.connection_manager import ConnectionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_clip()
    await init_SAM()

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

cwd = os.getcwd()

app.include_router(predict.router, prefix="/api")
app.include_router(score.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(refine.router, prefix="/api")
app.include_router(masks.router, prefix="/api")
app.include_router(inpaint.router, prefix="/api")
# app.include_router(prompts.router, prefix="/api", tags=["prompts"])
app.include_router(ws.router, prefix="/ws")
