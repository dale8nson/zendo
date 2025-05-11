from fastapi import FastAPI
from .routes import example

app = FastAPI()

app.include_router(example.router)

@app.get("/")
def read_root():
    return {"message": "ZendoAI Backend is running" }
