"""Shared paths and the meta.json contract between the Python and web sides."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WEB_MODELS = REPO / "web" / "public" / "models"

# COCO class ids the stock model uses for football.
COCO_ROLES = {0: "player", 32: "ball"}

# Role for each class name a football-specific dataset might use.
NAME_ROLES = {
    "player": "player",
    "goalkeeper": "goalkeeper",
    "referee": "referee",
    "ball": "ball",
    "football": "ball",
    "sports ball": "ball",
}


def roles_for(names: dict[int, str], coco: bool) -> dict[str, str]:
    """Maps class index -> app role; unrelated classes are left out."""
    if coco:
        return {str(k): v for k, v in COCO_ROLES.items()}
    return {str(i): NAME_ROLES[n.lower()] for i, n in names.items() if n.lower() in NAME_ROLES}


def write_meta(onnx_name: str, label: str, names: dict[int, str], imgsz: int, coco: bool) -> Path:
    """Writes meta.json, which tells the web app how to read the model output."""
    meta = {
        "file": onnx_name,
        "label": label,
        "input": imgsz,
        "names": [names[i] for i in sorted(names)],
        "roles": roles_for(names, coco),
    }
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    path = WEB_MODELS / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path
