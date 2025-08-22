from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .routes import ws, generate, refine, inpaint
from fastapi.responses import FileResponse
import os
from contextlib import asynccontextmanager
# from .services.clip_model import init_clip
# from .services.SAM import init_SAM

import torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # await init_clip()
    # await init_SAM()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

cwd = os.getcwd()

# app.include_router(predict.router, prefix="/api")
# app.include_router(score.router, prefix="/api")
# app.include_router(masks.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(refine.router, prefix="/api")
app.include_router(inpaint.router, prefix="/api")
app.include_router(ws.router, prefix="/ws")
