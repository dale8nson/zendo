from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import example, upload, metadata
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(upload.router, prefix="/api")

cwd = os.getcwd()
print(f"cwd: {cwd}")
app.mount("/assets", StaticFiles(directory=f"{cwd}/../../assets"), name="assets")


app.include_router(example.router)
app.include_router(metadata.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "ZendoAI Backend is running" }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("../../assets/zendo-logo-gradient-transparent-dark.svg")
