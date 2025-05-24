from fastapi import APIRouter
from app.services.metadata_handler import load_metadata

router = APIRouter()

@router.get("/metadata")
def get_metadata():
    return load_metadata()
