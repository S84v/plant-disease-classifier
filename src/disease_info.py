from pathlib import Path
import json

_METADATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "disease_metadata.json"
)

with open(_METADATA_PATH, "r", encoding="utf=8") as f:
    DISEASE_INFO = json.load(f)
