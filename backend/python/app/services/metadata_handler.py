import json
import os
from datetime import datetime
from typing import List, Dict
import sqlite3
from app.services.db import DB_PATH, get_connection


def log_prediction_to_db(filename: str, label: str, scores: list[float]):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO metadata (filename, label, scores) VALUES(?, ?, ?)",
            (filename, label, json.dumps(scores)),
        )

async def save_metadata_entry(entry: Dict, conn=None) -> None:
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO metadata (filename, original_filename, label) VALUES(?, ?, ?)",
        (entry["filename"], entry["original_filename"], entry["label"])
    )
    conn.commit()
    cursor.close()
