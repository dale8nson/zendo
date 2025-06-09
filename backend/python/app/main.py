from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import example, upload, metadata, predict, images, image
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from .services.db import init_db

init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(upload.router, prefix="/api")

cwd = os.getcwd()
print(f"cwd: {cwd}")
# app.mount("/assets", StaticFiles(directory=f"{cwd}/../../assets"), name="assets")
#
app.mount("/", StaticFiles(directory="static", html=True), name = "frontend")

uploads_dir = os.path.join(os.getcwd(), "app/uploads")
print(f"main.py: uploads_dir: {uploads_dir}")
if os.path.exists(uploads_dir):
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
else:
    print(f"[Warning] Uploads directory '{uploads_dir}' does not exist. Skipping upload mount.")

if os.path.exists("static/_next"):
    app.mount("/_next", StaticFiles(directory="static/_next"), name="next_data")

static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static_assets")
else:
    print(f"[Warning] Static directory '{static_dir}' does not exist. Skipping static mount.")


app.include_router(example.router)
app.include_router(metadata.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(image.router, prefix="/api")


# @app.get("/")
# def serve_index():
#     return StaticFiles(directory=os.path.join(cwd, "static")).lookup_path("index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("../../assets/zendo-logo-gradient-transparent-dark.svg")
