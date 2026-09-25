# Using the web app

## Running it

```bash
cd web
npm install
npm run dev        # development, http://localhost:5173
npm run build      # production build in web/dist
npm run preview    # serve the production build locally
```

`npm run dev` and `npm run build` first copy onnxruntime-web's WASM files
into `public/ort/`. If the model fails to load, check that folder exists.

## The screen

**01 Footage**, on the left.

- *Choose a clip*, or drag a video file onto the pitch. Anything your
  browser can play works: MP4 (H.264), WebM or MOV.
- *Analyse a browser tab* asks which tab to share. Pick the tab playing a
  highlight. The analysis follows it live, and the timecode shows **LIVE**.
- **Speed** slows playback so a slow computer can analyse more frames for
  each second of the match. The *Analysing* figure in the masthead shows how
  many frames per second are actually being analysed.
- **Markers / Numbers / Ball trail** toggle the overlay. Markers are corner
  brackets in the team's shirt colour. Numbers are tracking ids, not squad
  numbers. The yellow trail is the ball over the last 1.6 s, dashed where
  the path was filled in. A small yellow triangle under a player means they
  have the ball.
- Press **Space** to play or pause.

**02 The board**, top right.

- Possession split, with each side's colour taken from the detected kits.
  Click *Home* or *Away* to rename a team.
- Passes, balls won back, distance covered, and players on screen right now, per side.
- *Ball found in N% of analysed frames* tells you how much to trust the
  possession numbers. Below about 30%, treat them as rough.

**03 Shape**, bottom right.

- **Heatmap**: where each team (or the ball) has been, drawn as chalk on a
  pitch.
- **Pass network**: each player at their average position, sized by
  involvement, with lines as thick as the passes between them.
- **Timeline**: every pass and turnover. Click one to jump to two seconds
  before it (file mode only).

Exports: **Export JSON** gives the totals, per-player figures and all events.
**Events CSV** gives one event per row: `time_s, type, team, from_id, to_id`.
**Reset numbers** clears the totals without reloading the clip.

## Mapping the pitch

Until the pitch is mapped, heatmaps use the camera's view, and distance is
estimated from player height. To get real metres:

1. Pause on a frame where a known shape is fully visible.
2. Pick it under *Pitch mapping*: the whole pitch (four corner flags), the
   left or right penalty area, or the halfway line ends plus the centre
   circle's sides.
3. Click *Mark 4 points*, then click the four points on the video in the
   order the hint gives (clockwise from top-left, far side at the top).

The note beside the button changes to *Positions in metres*. Mapping holds
for one camera angle only. Wide broadcast shots pan, so re-map after a big
camera move. Clicks in a straight line are rejected with a message.

## Copy-protected streams

Paid apps such as Hotstar, SonyLIV, Netflix and Prime use DRM, and the
browser gives screen capture a black picture of them. If a shared tab stays
black for two seconds while playing, the app says so. Use highlight clips or
your own recordings instead. Trying to get around DRM would also break those
services' terms.

## Deploying

The build is a static site. Anything that serves files works, but two
headers make it faster:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

They allow multi-threaded WebAssembly. Without them the app still works, on
one thread. `web/vercel.json` sets them for Vercel: import the repo, set the
root directory to `web`, and deploy. GitHub Pages cannot set headers, so it
runs single-threaded there.

The first visit downloads about 40 MB (the model plus the WASM runtime).
After that, the browser cache serves both.

## Browser support

- **Chrome / Edge 113+**: WebGPU, fastest.
- **Firefox, Safari**: WebAssembly fallback, slower but complete.
- **Phones**: work for files, but are slow. Tab sharing is desktop only.

## Privacy

Frames are read from the `<video>` element into a canvas and passed to the
model on your device. Nothing is uploaded. The only network requests are for
the app's own files and Google Fonts.
