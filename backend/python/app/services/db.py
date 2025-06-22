import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPLOADS_DIR = HERE.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = UPLOADS_DIR / "zendoai.db"


def get_connection():
    """
    Returns a sqlite3.Connection to zendoai.db. If the file did not exist, this function will create it and run init_db() to create the metadata table.
    """

    print(f"{__file__}:DB_PATH: {DB_PATH}")
    is_new_db = not DB_PATH.exists()
    print(f"Is new DB: {is_new_db}")

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    if is_new_db:
        init_db(conn)
    return conn

def init_db(conn=None):
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    label TEXT,
    scores TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    width INTEGER,
    height INTEGER
    )"""
    )
    conn.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        source TEXT DEFAULT 'manual',
        is_active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT
        )
        """)
    conn.commit()
    cursor.close()
