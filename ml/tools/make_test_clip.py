"""Builds a synthetic broadcast-style clip with known ground truth.

Real people are cut out of the two sample photos that ship with ultralytics,
dressed in two kits, and moved over a perspective-drawn pitch while a ball is
passed between them on a fixed script. Useful for checking the web pipeline
end to end when no real footage is at hand.

    python tools/make_test_clip.py --out ../web/test-clip.webm

Writes <out> plus <out>.truth.json with the scripted passes and turnovers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import ultralytics
from ultralytics import YOLO

W, H, FPS, SECONDS = 1280, 720, 25, 24
ASSETS = Path(ultralytics.__file__).parent / "assets"
KITS = {0: (40, 30, 200), 1: (235, 235, 230), 2: (25, 25, 25)}  # BGR: red, white, referee black

# Pitch (metres) -> image: a broadcast camera on the halfway line looking across.
SRC = np.float32([[20, 5], [85, 5], [85, 63], [20, 63]])
DST = np.float32([[250, 190], [1030, 190], [1220, 690], [60, 690]])
H_PITCH = cv2.getPerspectiveTransform(SRC, DST)


def to_img(x: float, y: float) -> tuple[float, float]:
    p = H_PITCH @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def cutouts() -> list[np.ndarray]:
    """BGRA person cut-outs from the bundled sample images."""
    seg = YOLO("yolov8n-seg.pt")
    out = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(ASSETS / name))
        res = seg(img, classes=[0], verbose=False, retina_masks=True)[0]
        for m, b in zip(res.masks.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = b
            ratio = (y2 - y1) / max(1, x2 - x1)
            if (y2 - y1) < 200 or not 1.8 <= ratio <= 4.5 or x1 <= 2 or x2 >= img.shape[1] - 2:
                continue  # skip tiny, sprawling or edge-clipped people
            crop = img[y1:y2, x1:x2]
            alpha = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
            out.append(np.dstack([crop, alpha]))
    return out


def dress(sprite: np.ndarray, kit: tuple[int, int, int]) -> np.ndarray:
    """Recolours the torso band, keeping the photo's shading."""
    s = sprite.copy()
    h = s.shape[0]
    band = s[int(h * 0.18): int(h * 0.55)]
    lum = cv2.cvtColor(band[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    shade = np.clip(0.55 + lum * 0.7, 0, 1.2)[..., None]
    band[..., :3] = np.clip(np.array(kit, np.float32) * shade, 0, 255).astype(np.uint8)
    return s


def paste(frame: np.ndarray, sprite: np.ndarray, fx: float, fy: float, height: float) -> None:
    """Pastes a sprite so its feet stand on (fx, fy) in image pixels."""
    scale = height / sprite.shape[0]
    sp = cv2.resize(sprite, (max(1, int(sprite.shape[1] * scale)), int(height)), interpolation=cv2.INTER_AREA)
    x0, y0 = int(fx - sp.shape[1] / 2), int(fy - sp.shape[0])
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(W, x0 + sp.shape[1]), min(H, y0 + sp.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    part = sp[y1 - y0: y2 - y0, x1 - x0: x2 - x0]
    a = part[..., 3:4].astype(np.float32) / 255
    roi = frame[y1:y2, x1:x2]
    roi[:] = (part[..., :3] * a + roi * (1 - a)).astype(np.uint8)


def pitch_background() -> np.ndarray:
    top = np.full((H, W, 3), (38, 92, 44), np.uint8)  # run-off turf
    for i in range(14):  # mowing stripes along the length
        poly = np.int32([to_img(20 + i * 5, -30), to_img(25 + i * 5, -30), to_img(25 + i * 5, 90), to_img(20 + i * 5, 90)])
        cv2.fillPoly(top, [poly], (58, 128, 62) if i % 2 else (50, 116, 55))
    top[:170] = (40, 40, 44)  # stand
    line = lambda a, b: cv2.line(top, tuple(map(int, to_img(*a))), tuple(map(int, to_img(*b))), (225, 230, 228), 2, cv2.LINE_AA)
    line((0, 0), (105, 0)); line((0, 68), (105, 68)); line((52.5, 0), (52.5, 68))
    circle = [to_img(52.5 + 9.15 * math.cos(t), 34 + 9.15 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 60)]
    cv2.polylines(top, [np.int32(circle)], True, (225, 230, 228), 2, cv2.LINE_AA)
    return top


def script():
    """Scripted possession: (holder index, seconds) — ball travels 0.6 s between holders."""
    return [(0, 2.0), (1, 1.6), (2, 1.6), (6, 2.0), (7, 1.6), (8, 1.6), (3, 2.0), (4, 1.6), (0, 2.0), (6, 2.0)]


def main(out: Path) -> None:
    sprites = cutouts()
    teams = [0] * 5 + [1] * 5 + [2]
    homes = [(35, 20), (45, 30), (40, 48), (55, 16), (60, 40), (70, 22), (65, 34), (72, 50), (50, 55), (80, 36), (52, 30)]
    kitted = [dress(sprites[i % len(sprites)], KITS[teams[i]]) for i in range(len(teams))]
    bg = pitch_background()

    # Ball timeline: hold at a player's feet, then fly to the next one.
    segs, t, truth = [], 0.0, []
    plan = script()
    for i, (who, hold) in enumerate(plan):
        segs.append(("hold", t, t + hold, who, who))
        t += hold
        if i + 1 < len(plan):
            nxt = plan[i + 1][0]
            segs.append(("fly", t, t + 0.6, who, nxt))
            kind = "pass" if teams[who] == teams[nxt] else "turnover"
            truth.append({"t": round(t + 0.6, 2), "type": kind, "team": teams[nxt]})
            t += 0.6

    writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libvpx-vp9", quality=None,
                                         output_params=["-b:v", "3M", "-pix_fmt", "yuv420p"])
    writer.send(None)
    for f in range(FPS * SECONDS):
        t = f / FPS
        pos = [(hx + 4 * math.sin(t * 0.7 + i), hy + 3 * math.cos(t * 0.9 + i * 1.3)) for i, (hx, hy) in enumerate(homes)]
        frame = bg.copy()
        seg = next((s for s in segs if s[1] <= t < s[2]), segs[-1])
        a, b = pos[seg[3]], pos[seg[4]]
        k = 0.0 if seg[0] == "hold" else (t - seg[1]) / (seg[2] - seg[1])
        bx, by = a[0] + (b[0] - a[0]) * k + 0.6, a[1] + (b[1] - a[1]) * k + 0.3
        lift = math.sin(k * math.pi) * 18 if seg[0] == "fly" else 0
        for i in sorted(range(len(pos)), key=lambda i: pos[i][1]):  # far players first
            fx, fy = to_img(*pos[i])
            paste(frame, kitted[i], fx, fy, 40 + (fy - 190) / 500 * 75)
        ix, iy = to_img(bx, by)
        r = max(4, int(4 + (iy - 190) / 500 * 5))
        cv2.circle(frame, (int(ix), int(iy - lift)), r, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(ix), int(iy - lift)), max(1, r // 3), (30, 30, 30), -1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(frame[..., ::-1]))
    writer.close()
    Path(str(out) + ".truth.json").write_text(json.dumps({"events": truth, "pitch_to_image": H_PITCH.tolist()}, indent=2))
    print(f"wrote {out} with {len(truth)} scripted events")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("test-clip.webm"))
    main(ap.parse_args().out)
