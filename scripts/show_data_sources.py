#!/usr/bin/env python3
"""Print registered upstream products and their official records."""

import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "configs/data_sources.json"
sources = json.loads(path.read_text())
for name, item in sources.items():
    print(f"{name:16s} {item.get('version', '')}")
    print(f"  role:    {item['role']}")
    if "record" in item:
        print(f"  record:  {item['record']}")
    print(f"  license: {item['license']}")
