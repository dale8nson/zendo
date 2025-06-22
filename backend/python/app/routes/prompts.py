from fastapi import APIRouter
from app.services.prompts import prompts, add, delete, update

router = APIRouter()
DB_PATH = "app/uploads/metadata.db"

@router.get("/prompts")
def list_prompts():
    return prompts()

@router.post("/prompts")
def add_prompt(text: str, source: str = "manual"):
    return add(text, source)

@router.delete("/prompts/{prompt_id}")
def remove_prompt(prompt_id: int):
    return delete(prompt_id)

@router.put("/prompts/{prompt_id}")
def update_prompt(prompt_id: int, text: str, source: str = "manual"):
    return update(prompt_id, text, source)
