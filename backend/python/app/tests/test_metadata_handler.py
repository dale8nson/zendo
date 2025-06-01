import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import sqlite3
from app.services.metadata_handler import save_metadata_entry
from app.services.db import init_db

@pytest.fixture
def test_db():
    conn = sqlite3.connect(':memory:')
    init_db(conn)
    yield conn
    conn.close()

def test_save_metadata_entry(test_db):
    entry = {
    "filename": "test_image.webp",
    "label": "wall",
    "timestamp": "2025-07-31T12:00:00"
    }

    save_metadata_entry(entry, conn=test_db)

    cursor = test_db.cursor()
    cursor.execute("SELECT filename, label, timestamp FROM predictions WHERE filename = ?", (entry["filename"],))
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == entry["filename"]
    assert row[1] == entry["label"]
    assert isinstance(row[2], str)
    assert len(row[2]) >= 19
