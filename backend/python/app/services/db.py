import sqlite3
from pathlib import Path

DB_PATH = Path("app/uploads/zendoai.db")

def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

def init_db(conn=None):
    if conn is None:
        conn = get_connection()

    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT,
    scores TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
