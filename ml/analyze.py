"""Offline analysis: runs detection + ByteTrack over a whole video file.

    python analyze.py match.mp4
    python analyze.py match.mp4 --weights runs/detect/football/weights/best.pt --stride 2

Handy for long matches (no browser tab to keep open) and as a reference to
compare the web app's tracking against. Output: per-frame boxes with track
ids, plus how often the ball was found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from types import SimpleNamespace

import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from common import COCO_ROLES, NAME_ROLES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()

    model = YOLO(args.weights)
    roles = COCO_ROLES if len(model.names) == 80 else {
        i: NAME_ROLES[n.lower()] for i, n in model.names.items() if n.lower() in NAME_ROLES
    }
    fps = cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FPS) or 25.0

    # Only people go through ByteTrack: it would drop the low-confidence ball
    # detections that possession tracking still needs.
    cfg = SimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.track_buffer = max(10, round(cfg.track_buffer / args.stride))  # keep ~1 s of memory
    tracker = BYTETracker(cfg)
    ball_cls = [c for c, r in roles.items() if r == "ball"]

    frames, ball_frames = [], 0
    stream = model.predict(str(args.video), stream=True, conf=args.conf, classes=list(roles),
                           vid_stride=args.stride, verbose=False)
    for i, res in enumerate(stream):
        boxes = res.boxes.cpu()
        is_ball = sum((boxes.cls == c) for c in ball_cls).bool() if ball_cls else boxes.cls < 0
        tracked = tracker.update(boxes[~is_ball], res.orig_img)
        players = [
            {"id": int(t[4]), "role": roles[int(t[6])], "box": [round(float(v), 1) for v in t[:4]]}
            for t in tracked
        ]
        balls = boxes[is_ball]
        ball = [round(v, 1) for v in balls.xyxy[int(balls.conf.argmax())].tolist()] if len(balls) else None
        ball_frames += ball is not None
        frames.append({"t": round(i * args.stride / fps, 3), "players": players, "ball": ball})

    summary = {
        "video": args.video.name,
        "fps": fps,
        "frames_analysed": len(frames),
        "ball_found_share": round(ball_frames / max(1, len(frames)), 3),
        "unique_tracks": len({p["id"] for f in frames for p in f["players"]}),
    }
    out = args.out or args.video.with_suffix(".analysis.json")
    out.write_text(json.dumps({"summary": summary, "frames": frames}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
