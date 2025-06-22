from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import upload, metadata, predict, images, image, prompts, caption
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from .services.db import init_db, UPLOADS_DIR
from contextlib import asynccontextmanager
from .services.clip_model import init_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_model()
    yield

app = FastAPI(lifespan=lifespan)

init_db()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

cwd = os.getcwd()
print(f"cwd: {cwd}")

# uploads_dir = os.path.join(cwd, "app/uploads")
print(f"main.py: uploads_dir: {UPLOADS_DIR}")
if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
    print(f"uploads directory mounted at: {UPLOADS_DIR}")
else:
    print(f"[Warning] Uploads directory '{UPLOADS_DIR}' does not exist. Skipping upload mount.")

# app.include_router(example.router)
app.include_router(upload.router, prefix="/api")
app.include_router(metadata.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(image.router, prefix="/api")
app.include_router(caption.router, prefix="/api")
app.include_router(prompts.router, prefix="/api", tags=["prompts"])


static_dir = "static"
if os.path.exists(static_dir):
    # app.mount("/static", StaticFiles(directory=static_dir), name="static_assets")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

else:
    print(f"[Warning] Static directory '{static_dir}' does not exist. Skipping static mount.")

if os.path.exists("static/_next"):
    app.mount("/_next", StaticFiles(directory="static/_next"), name="next_data")
# if os.path.exists(static_dir):
    # app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
# else:
    # @app.get("/")
    # def serve_index():
        # return StaticFiles(directory=os.path.join(cwd, "static")).lookup_path("index.html")
    # execv(os.path.join(cwd, "..", "..", "frontend"), ["npm", "run", "dev"])

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(cwd, "assets/zendo-logo-gradient-transparent-dark.svg"))



# if __name__ == "__main__":
