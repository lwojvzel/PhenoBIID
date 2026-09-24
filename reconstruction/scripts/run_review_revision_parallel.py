from __future__ import annotations
import json
def atomic_json(path, values):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(path)
