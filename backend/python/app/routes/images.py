from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.db import get_connection, UPLOADS_DIR
from pathlib import Path
import base64
from typing import Dict, List, Any
import os

router = APIRouter()

@router.get("/images")
def get_images():

    print("Files in upload dir:", os.listdir(UPLOADS_DIR))
    conn = get_connection()
    cursor = conn.cursor()
    cursor = cursor.execute("""
        SELECT * FROM metadata WHERE timestamp = (
            SELECT MAX(timestamp) FROM metadata as m2
            WHERE m2.filename = metadata.filename
        )
        """)
    metadata = cursor.fetchall()
    print(f"metadata: {metadata}")
    conn.close()

    results: List[Dict[str, Any]] = []

    print(f"Fetched {len(metadata)} rows from DB")

    for datum in metadata:
        id_, filename, original_filename, label, prediction, timestamp = datum
        filename = Path(str(filename))
        path = Path(UPLOADS_DIR) / filename
        print(f"path: {path}")
        if not path.exists() or not path.is_file():
            continue

        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if path.is_file():
            with open(path, "rb") as f:
                image_bytes = f.read()
                encoded = base64.b64encode(image_bytes).decode('utf-8')
            results.append({
                        "id": id_,
                        "filename": str(filename),
                        "original_filename": original_filename,
                        "label": label,
                        "prediction": prediction,
                        "timestamp": timestamp,
                        "image_data": encoded
                    })
            print(f"len(results): {len(results)}")

    return JSONResponse(content=results)
