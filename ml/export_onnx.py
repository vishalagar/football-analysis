"""Exports a YOLOv8 checkpoint to ONNX for the browser.

    python export_onnx.py                      # stock COCO yolov8n
    python export_onnx.py --weights runs/detect/football/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from common import WEB_MODELS, write_meta


def export(weights: str, imgsz: int) -> Path:
    model = YOLO(weights)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False))
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    dst = WEB_MODELS / "model.onnx"
    shutil.copy(onnx_path, dst)
    coco = len(model.names) == 80
    label = f"{Path(weights).stem} · {'COCO' if coco else 'football'}"
    meta = write_meta(dst.name, label, model.names, imgsz, coco)
    print(f"model -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)\nmeta  -> {meta}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    export(args.weights, args.imgsz)
