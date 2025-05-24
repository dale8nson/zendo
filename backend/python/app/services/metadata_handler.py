import json
import os
from datetime import datetime
from typing import List, Dict

METADATA_FILE = "app/uploads/metadata.json"

def load_metadata() -> List[Dict]:
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata_entry(entry: Dict) -> None:
    metadata = load_metadata()
    entry["timestamp"] = datetime.utcnow().isoformat()
    metadata.append(entry)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
