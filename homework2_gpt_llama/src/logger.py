import json
import os
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_jsonl(filepath: str, record: dict):
    ensure_dir(os.path.dirname(filepath))
    record = dict(record)
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
