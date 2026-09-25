"""Fine-tunes YOLOv8n on football footage, then exports it for the browser.

    python train.py --data data/football.yaml --epochs 60
    python train.py --data data/football.yaml --device 0 --imgsz 1280   # on a GPU

The stock COCO model sees a broadcast ball in few frames. A football dataset
with a dedicated ball class, trained at a larger image size, is the biggest
single improvement to the app. Run this on Colab or any CUDA machine; CPU
training works but takes hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from export_onnx import export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/football.yaml")
    ap.add_argument("--base", default="yolov8n.pt", help="starting checkpoint")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280, help="train size; larger helps the ball")
    ap.add_argument("--export-size", type=int, default=640, help="browser input size")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu' to force CPU")
    ap.add_argument("--name", default="football")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # Broadcast footage: no vertical flips, mild colour shifts so kit colours survive.
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.01,
        mosaic=1.0,
        close_mosaic=10,
        patience=20,
    )
    best = Path(model.trainer.best)
    print(f"best checkpoint: {best}")
    export(str(best), args.export_size)


if __name__ == "__main__":
    main()
