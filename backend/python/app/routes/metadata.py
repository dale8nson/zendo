import sqlite3
from fastapi import APIRouter
from app.services.db import DB_PATH

router = APIRouter()

@router.get("/metadata")
def get_metadata():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
