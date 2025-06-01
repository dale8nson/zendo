import json
import os
from datetime import datetime
from typing import List, Dict
import sqlite3
from app.services.db import DB_PATH, get_connection

def log_prediction_to_db(filename: str, label: str, scores: list[float]):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO predictions (filename, label, scores) VALUES(?, ?, ?", (filename, label, json.dumps(scores)))


METADATA_FILE = "app/uploads/metadata.json"

def load_metadata() -> List[Dict]:
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata_entry(entry: Dict, conn=None) -> None:
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (filename, label) VALUES(?, ?)", (entry["filename"], entry["label"]))
    conn.commit()
