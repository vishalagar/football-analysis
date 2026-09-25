# Architecture

Football Vision has two halves that meet at one file format.

```
 ml/ (Python)                          web/ (browser)
 ───────────────                       ──────────────────────────────────────
 train.py ─┐                           <video> ─► Detector ─► Tracker ─► Teams
 yolov8n.pt├► export_onnx.py ─► model.onnx        │            │          │
 dataset  ─┘                    meta.json         ▼            ▼          ▼
                                                  Ball ──► Possession ─► Stats ─► board
 analyze.py (offline reference)                                           │
                                                                          ▼
                                                            heatmap · network · timeline
```

`model.onnx` plus `meta.json` is the whole interface. The web app never
imports Python and the Python side never needs the web app.

## The per-frame pipeline

`web/src/main.ts` runs a `requestAnimationFrame` loop. Drawing happens on
every frame. Analysis runs only when the previous inference has finished and
enough media time has passed. The rate adapts to the device: the minimum
interval is about 1.15 × the last inference time, capped at 15 per second.

For each analysed frame:

1. **Cut check** (`source.ts › CutDetector`). A 16-bin brightness histogram is
   compared with the previous frame's. A large jump means the broadcast
   changed camera, so tracks, the ball trail and possession are reset. Totals
   are kept.
2. **Detection** (`detector.ts`). The frame is letterboxed into the model's
   square input and run through ONNX Runtime Web. `yolo.ts` decodes the raw
   `[1, 4 + classes, anchors]` output, maps class indices to roles (`player`,
   `goalkeeper`, `referee`, `ball`) using `meta.json`, and applies per-role
   thresholds and non-max suppression.
3. **Ball focus**. If no ball was found and it was seen within the last
   second, the model runs again on a half-size crop around that spot. That
   is a 2× zoom, which makes a small ball much easier to find.
4. **Tracking** (`tracker.ts`). A SORT-style tracker without a Kalman
   filter: constant-velocity prediction, greedy IoU matching, then a second
   pass on centre distance for fast movers whose boxes no longer overlap.
   Tracks need 3 hits before they count and are dropped after 12 misses.
5. **Teams** (`teams.ts`). The torso of each tracked player is sampled
   (turf-green pixels skipped) and smoothed per track. After about 30 samples,
   k-means with k = 3 runs. The two biggest clusters are the teams. The third
   counts as officials only if it is a real share of samples *and* a clearly
   different colour. Otherwise it's treated as shading and k = 2 is used.
   Colours far from every centre are labelled as outliers.
6. **Ball** (`ball.ts`). One ball is kept. Among the candidates it picks the
   best score close to the last position, rejects implausible jumps, and
   fills gaps of up to 0.5 s by linear interpolation. Filled points are drawn
   dashed.
7. **Possession** (`events.ts`). The player whose feet are closest to the
   ball, within 0.6 × their height, becomes a candidate. After 0.12 s they
   hold the ball. When the holder changes, a *pass* is recorded if the old
   and new holder are team-mates, and a *turnover* (ball won back) if not.
   The last holder is remembered for 6 s, so a long ball still links up.
8. **Stats** (`stats.ts`). Possession time is credited to the team in
   control. Distance is added per player: in metres through the homography
   once the pitch is calibrated, otherwise by using player height (about
   1.8 m) as a local ruler. Anything faster than 11 m/s is treated as a
   tracking glitch and ignored. Positions feed 5 m heat grids, and passes
   feed the network edges.

The board refreshes four times a second. It doesn't need more, and that
keeps the main thread free for inference.

## Pitch mapping

`pitch.ts` solves a 3×3 homography from four clicked image points to four
known pitch points (whole pitch, either penalty area, or the halfway-line
ends plus the sides of the centre circle). It uses Gaussian elimination on
the standard 8-unknown system. Collinear clicks are rejected. Foot points
(bottom-centre of each box) are projected, because that is where a player
touches the ground.

## Why these choices

- **Vanilla TypeScript, no framework.** The UI is one screen of mostly
  canvas and SVG. A framework would add weight without making the analysis
  code any clearer.
- **Tracker without a Kalman filter.** At 5–15 analysed frames per second,
  constant velocity plus a distance pass performs about as well, and each
  step is readable in one screen of code.
- **Colour clustering instead of a re-ID network.** Kit colour is the one
  thing that reliably separates teams, and it is cheap.
- **ONNX Runtime Web, not TensorFlow.js.** ultralytics exports ONNX
  directly, and ORT has WebGPU with a WASM fallback.

## Build details worth knowing

- onnxruntime-web is loaded through its `onnxruntime-web-use-extern-wasm`
  export condition (see `vite.config.ts`). Its WASM files are served from
  `public/ort/`, copied there by `scripts/copy-ort.mjs`. Bundling them
  instead makes ORT's worker threads import the app's own chunks, which
  touch `document` and crash.
- The dev server, preview and `vercel.json` send COOP/COEP headers. That
  enables cross-origin isolation, which multi-threaded WASM needs.
  `credentialless` keeps Google Fonts working.

## Testing

- `web/tests/*.test.ts` covers the tracker, team clustering, possession and
  pass logic, stats, homography, YOLO decoding, NMS and cut detection.
- `ml/tests/test_export.py` checks the shipped ONNX model has the input and
  output shapes the web decoder expects, and that `meta.json` roles point at
  real classes.
- `ml/tools/make_test_clip.py` builds a clip with scripted passes (real
  people cut out of the ultralytics sample photos, on a perspective pitch)
  for end-to-end checks in a real browser.
