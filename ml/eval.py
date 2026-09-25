"""Scores a checkpoint or ONNX model on the validation split.

    python eval.py --weights runs/detect/football/weights/best.pt

Prints mAP50 per class and ball recall, the number that matters most for
possession and pass detection.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/football.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    m = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, verbose=False)
    names = m.names
    print(f"{'class':<12}{'mAP50':>8}{'recall':>8}")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"{names[c]:<12}{m.box.ap50[i]:>8.3f}{m.box.r[i]:>8.3f}")
    print(f"{'all':<12}{m.box.map50:>8.3f}")


if __name__ == "__main__":
    main()
