# The model

The web app runs any YOLOv8-style detector exported to ONNX. Two files in
`web/public/models/` describe it:

| File | What it is |
|---|---|
| `model.onnx` | The network. Input `[1, 3, S, S]` float RGB in 0–1, output `[1, 4 + C, anchors]`. |
| `meta.json` | How to read it: input size, class names, and which classes matter. |

## meta.json

```json
{
  "file": "model.onnx",
  "label": "yolov8n · COCO",
  "input": 640,
  "names": ["person", "bicycle", "…"],
  "roles": { "0": "player", "32": "ball" },
  "thresholds": { "ball": 0.12 }
}
```

- `roles` maps class index to the part it plays in the app: `player`,
  `goalkeeper`, `referee` or `ball`. Classes that aren't listed are ignored.
- `thresholds` is optional and overrides the defaults per role (people 0.35,
  ball 0.12). A low ball threshold is deliberate: a missed ball costs more
  than a false one, because the ball tracker rejects impossible jumps anyway.
- `label` is what the masthead shows.

`ml/export_onnx.py` writes both files. You should never have to edit them by
hand.

## v1: the stock COCO model

The repo ships YOLOv8n trained on COCO (12.9 MB, 640 px input). COCO has a
`person` class (0) and a `sports ball` class (32), which is enough to get
going.

What to expect:

- **Players:** reliable once they are at least 30–40 px tall.
- **Ball:** weak. COCO's balls are mostly close-ups, while a broadcast ball
  is a handful of pixels. On the synthetic test clip, full-frame recall at
  the app's threshold is about 5%. Rerunning on a 2× zoomed crop raises it to
  about 27%. The app's ball-focus pass does this automatically, around the
  last known position.
- **Referees and goalkeepers** are just `person`. The team clustering flags
  referees by kit colour instead.

Re-export any time:

```bash
cd ml && python export_onnx.py
```

## v2: fine-tune on football footage

This is the upgrade that matters most. A dataset with a real `ball` class,
trained at a higher resolution, fixes most of the ball problem and also
separates goalkeepers and referees properly.

1. **Get data.** Any YOLO-format football dataset works. Roboflow Universe's
   *football-players-detection* has `ball`, `goalkeeper`, `player` and
   `referee` classes from broadcast footage. Download it in "YOLOv8" format
   into `ml/datasets/`, then check `path:` and `names:` in
   `ml/data/football.yaml`.
2. **Train** on a GPU (Colab's free T4 is enough):

   ```bash
   cd ml
   python train.py --data data/football.yaml --epochs 60 --imgsz 1280 --device 0
   ```

   Training at 1280 while exporting at 640 is intentional. The model learns
   small balls from sharp examples and still runs fast in the browser. If
   your users have fast machines, export at 960 with `--export-size 960`.
3. **Check it:**

   ```bash
   python eval.py --weights runs/detect/football/weights/best.pt
   ```

   Look at ball recall first. Player mAP is usually fine from the start.
4. **Ship it.** `train.py` already calls the exporter, so `model.onnx` and
   `meta.json` are replaced. Reload the web app; no code changes needed.
   `roles` is filled from the class names using `NAME_ROLES` in
   `ml/common.py`.

Training settings in `train.py` suited to broadcast footage:

- no vertical flips (the sky is never below the pitch)
- light hue shift only, so kit colours stay true
- mosaic augmentation, switched off for the last 10 epochs

## Size and speed

YOLOv8n at 640 px is 13 MB. Measured in development, headless Chromium on a
4-core cloud machine with no GPU (WebAssembly) managed 1–3 analysed frames
per second, including the ball-focus pass. A laptop with WebGPU should be
several times faster, but that hasn't been measured here. The *Analysing*
figure in the masthead shows the real rate on your machine.

Rules of thumb:

- A larger export size (`--export-size 960`) finds more balls, at roughly
  twice the cost per frame.
- A larger model (YOLOv8s, 45 MB) is more accurate and 2–3× slower.
- On a slow machine, set Speed to 0.5× or 0.25× so more frames get analysed
  for each second of play.

## Offline analysis

`ml/analyze.py` runs the same model over a whole file with ultralytics'
ByteTrack and writes per-frame boxes, track ids and the ball position to
JSON. Only people go through ByteTrack, because it would drop
low-confidence ball detections that possession still needs. Use it for full
matches, or as a reference when checking the web tracker.

## Licence

Ultralytics YOLOv8 code and weights are AGPL-3.0, and so are models
fine-tuned from them. For a public deployment, keep the source available, or
buy an Ultralytics enterprise licence, or swap in a detector with a licence
you prefer. Anything that exports to the same ONNX output layout works
unchanged.
