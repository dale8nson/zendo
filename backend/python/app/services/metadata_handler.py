import json
from typing import Dict
import sqlite3
from app.services.db import DB_PATH, get_connection


def log_prediction_to_db(filename: str, original_filename: str, label: str, scores: list[float], width: int, height: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO metadata (filename, original_filename, scores, width, height) VALUES(?, ?, ?, ?, ?)",
            (filename, original_filename, label, json.dumps(scores), width, height),
        )

async def save_metadata_entry(entry: Dict, conn=None) -> None:
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO metadata (filename, original_filename, width, height, collection) VALUES(?, ?, ?, ?, ?)",
        (entry["filename"], entry["original_filename"], entry["width"], entry["height"], entry["collection"])
    )
    conn.commit()
    cursor.close()
