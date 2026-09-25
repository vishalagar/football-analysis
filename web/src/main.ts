import './styles.css';
import { BallTrack } from './ball';
import { Detector } from './detector';
import { Possession } from './events';
import { foot } from './geometry';
import { clampToPitch, homography, PITCH, project, REFERENCES, type Homography } from './pitch';
import { drawHeat, networkSvg } from './render/boards';
import { drawOverlay, fitView, readableOn } from './render/overlay';
import { pitchSvg } from './render/pitchSvg';
import { CutDetector, DrmWatch, loadFile, loadTab, stopStream, type SourceKind } from './source';
import { MatchStats, type Sample } from './stats';
import { retry } from './retry';
import { jerseyColour, TeamModel } from './teams';
import { Tracker, type Track } from './tracker';
import type { Box, MatchEvent, Point, Rgb } from './types';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');
const screen = $('screen');

const detector = new Detector();
const tracker = new Tracker();
const teams = new TeamModel();
const ball = new BallTrack();
const possession = new Possession();
const cuts = new CutDetector();
const drm = new DrmWatch();
let stats = new MatchStats();
let timeline: MatchEvent[] = [];

const MODEL_ATTEMPTS = 3;
let ready = false;
let modelFailed = false;
let source: SourceKind | null = null;
let busy = false;
let lastDetT = -1;
let tabStart = 0;
let H: Homography | null = null;
let calib: Point[] | null = null;
let calibRef = 'full';
let detTimes: number[] = [];
const opts = { markers: true, ids: true, trail: true };
const sampler = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
sampler.canvas.width = sampler.canvas.height = 10;

const CHALK: Rgb = [236, 234, 224];
const rgbCss = ([r, g, b]: Rgb) => `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
const teamRgb = (i: 0 | 1): Rgb => teams.centres[i] ?? CHALK;
const kitCss = (tr: Track) =>
  tr.team === 0 || tr.team === 1 ? rgbCss(teamRgb(tr.team)) : tr.team === 2 ? '#8A8F86' : rgbCss(CHALK);

// ---------- analysis ----------

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const now = () => (source === 'tab' ? performance.now() / 1000 - tabStart : video.currentTime);

function toPitch(p: Point): Point {
  if (H) return clampToPitch(project(H, p));
  return { x: (p.x / video.videoWidth) * PITCH.length, y: (p.y / video.videoHeight) * PITCH.width };
}

function kitAt(b: Box): Rgb | null {
  if (b.h < 24) return null;
  // Torso only: head, arms and shorts dilute the kit colour.
  sampler.drawImage(video, b.x + b.w * 0.3, b.y + b.h * 0.2, b.w * 0.4, b.h * 0.28, 0, 0, 10, 10);
  return jerseyColour(sampler.getImageData(0, 0, 10, 10).data);
}

function sampleKit(tr: Track): void {
  const c = kitAt(tr);
  if (!c) return;
  tr.colour = tr.colour ? (tr.colour.map((v, i) => v * 0.8 + c[i] * 0.2) as Rgb) : c;
  if (tr.hits % 2 === 0) teams.add(tr.colour);
}

function resetMotion(): void {
  tracker.reset();
  ball.reset();
  possession.reset();
  cuts.reset();
  stats.forgetPositions();
  lastDetT = -1;
}

async function analyse(t: number): Promise<void> {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (cuts.isCut(video)) resetMotion();
  const dets = await detector.detect(video, vw, vh);
  ball.frameW = vw;
  const lastBall = ball.last;
  if (!dets.some((d) => d.kind === 'ball') && lastBall && t - lastBall.t < 1) {
    // Ball focus: a small ball is often missed at full-frame scale, so look
    // again at 2× zoom around where it was last seen.
    const w = vw / 2, h = vh / 2;
    const roi = { x: clamp(lastBall.x - w / 2, 0, vw - w), y: clamp(lastBall.y - h / 2, 0, vh - h), w, h };
    dets.push(...(await detector.detect(video, vw, vh, roi)).filter((d) => d.kind === 'ball'));
  }
  const tracks = tracker.update(dets, t, kitAt);
  for (const tr of tracks) {
    sampleKit(tr);
    tr.team = tr.kind === 'referee' ? 2 : teams.classify(tr.colour);
  }
  ball.update(dets, t);
  const bp = ball.current(t);
  const ev = possession.update(t, bp, tracks.filter((tr) => tr.kind !== 'referee'));
  const samples: Sample[] = tracks.map((tr) => {
    const f = foot(tr);
    return { id: tr.id, team: tr.team, pos: toPitch(f), px: f, mpp: 1.8 / tr.h };
  });
  stats.step(t, lastDetT < 0 ? 0 : t - lastDetT, samples, bp ? toPitch(bp) : null, possession.team(t), !!H);
  if (ev) {
    stats.event(ev);
    timeline.unshift(ev);
  }
  lastDetT = t;
  detTimes.push(performance.now());
}

function frame(): void {
  requestAnimationFrame(frame);
  if (!ready || video.readyState < 2 || !video.videoWidth) return;
  const t = now();
  if (source === 'tab' && drm.check(video, performance.now())) showNotice('drm');
  if (lastDetT >= 0 && t < lastDetT - 0.05) resetMotion();
  const interval = Math.max(1 / 15, (detector.lastMs * 1.15) / 1000);
  if (!busy && !video.paused && (lastDetT < 0 || t - lastDetT >= interval)) {
    busy = true;
    analyse(t).catch(console.error).finally(() => (busy = false));
  }
  const view = fitView(overlay, video.videoWidth, video.videoHeight);
  drawOverlay(overlay, view, tracker.predicted(t), kitCss, ball.trail, t, possession.holder?.id ?? null, calib ?? [], opts);
}

// ---------- panel ----------

const fmt = (s: number) => {
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

function setNum(id: string, v: string): void {
  const el = $(id);
  if (el.textContent === v) return;
  el.textContent = v;
  el.classList.remove('tick');
  void el.offsetWidth;
  el.classList.add('tick');
}

let heatView: 0 | 1 | 'ball' = 0;
let netView: 0 | 1 = 0;
let tab = 'heat';

function panel(): void {
  const [pa, pb] = stats.possessionShare();
  setNum('pos-a', String(pa));
  setNum('pos-b', String(pb));
  $('split-a').style.width = `${pa}%`;
  for (const i of [0, 1] as const) {
    const c = teams.ready ? rgbCss(teamRgb(i)) : i ? 'rgb(58,64,60)' : 'rgb(92,98,92)';
    document.documentElement.style.setProperty(`--kit-${i ? 'b' : 'a'}`, c);
    document.documentElement.style.setProperty(`--kit-${i ? 'b' : 'a'}-ink`, readableOn(c));
  }
  const rows: Record<string, [string, string]> = {
    passes: [String(stats.passes[0]), String(stats.passes[1])],
    won: [String(stats.turnoversWon[0]), String(stats.turnoversWon[1])],
    dist: [(stats.teamMetres(0) / 1000).toFixed(2), (stats.teamMetres(1) / 1000).toFixed(2)],
    seen: [String(stats.onScreen[0]), String(stats.onScreen[1])],
  };
  for (const [k, [a, b]] of Object.entries(rows)) {
    setNum(`s-${k}-a`, a);
    setNum(`s-${k}-b`, b);
  }
  const ballPct = stats.frames ? Math.round((stats.ballFrames / stats.frames) * 100) : 0;
  $('ball-seen').textContent = stats.frames ? `Ball found in ${ballPct}% of analysed frames` : 'Waiting for play';
  $('kits').textContent = teams.ready ? 'Kits identified' : 'Reading kits…';

  const cut = performance.now() - 2000;
  detTimes = detTimes.filter((x) => x > cut);
  $('m-rate').textContent = ready ? `${(detTimes.length / 2).toFixed(1)} frames/s` : '—';
  $('tc').textContent = source === 'tab' ? 'LIVE' : `${fmt(video.currentTime || 0)} / ${fmt(video.duration || 0)}`;
  if (source === 'file' && video.duration) $<HTMLInputElement>('scrub').value = String((video.currentTime / video.duration) * 1000);

  if (tab === 'heat') {
    const grid = heatView === 'ball' ? stats.ballHeat : stats.heat[heatView];
    drawHeat($<HTMLCanvasElement>('heat'), grid, heatView === 'ball' ? [242, 194, 48] : teamRgb(heatView));
  } else if (tab === 'net') {
    $('net').innerHTML = pitchSvg(networkSvg(stats, netView, rgbCss(teamRgb(netView))));
  } else {
    renderTimeline();
  }
  $('pitch-note').textContent = H ? 'Positions in metres — pitch calibrated' : 'Camera view — calibrate the pitch for true positions';
}

function renderTimeline(): void {
  const ol = $('timeline');
  if (!timeline.length) {
    ol.innerHTML = '<li class="empty-row">Passes and turnovers appear here as they happen.</li>';
    return;
  }
  ol.innerHTML = timeline.slice(0, 200).map((e) => `
    <li><button data-t="${e.t}" class="ev ev--${e.team ? 'b' : 'a'}">
      <time>${fmt(e.t)}</time><span class="ev__type">${e.type === 'pass' ? 'Pass' : 'Won ball'}</span>
      <span class="ev__who">#${e.from} <span aria-hidden="true">→</span> #${e.to}</span></button></li>`).join('');
}

// ---------- notices ----------

const NOTICES: Record<string, string> = {
  drm: 'This stream is copy-protected, so the browser only hands us a black picture. Paid apps like Hotstar, SonyLIV and Netflix do this. Use highlight clips (YouTube, FanCode) or a video file instead.',
  tab: 'Tab sharing was cancelled or is not available in this browser.',
  model: 'The detection model did not load, most likely a dropped connection. Reload the page to try again.',
  degenerate: 'Those four points are in a line, so the pitch cannot be mapped. Try again with points that form a box.',
};

function showNotice(key: keyof typeof NOTICES | null): void {
  const n = $('notice');
  n.hidden = !key;
  if (key) n.textContent = NOTICES[key];
}

// ---------- wiring ----------

function started(kind: SourceKind, label: string): void {
  source = kind;
  stats = new MatchStats();
  timeline = [];
  teams.reset();
  drm.reset();
  H = null;
  resetMotion();
  // A failed model load must stay visible, or the app just sits on "Loading…".
  showNotice(modelFailed ? 'model' : null);
  document.body.dataset.state = 'loaded';
  document.body.dataset.source = kind;
  $('m-clip').textContent = label;
  $<HTMLInputElement>('scrub').disabled = kind === 'tab';
  $<HTMLSelectElement>('speed').disabled = kind === 'tab';
  video.playbackRate = kind === 'tab' ? 1 : Number($<HTMLSelectElement>('speed').value);
}

$<HTMLInputElement>('file').addEventListener('change', (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  loadFile(video, f);
  started('file', f.name);
});

screen.addEventListener('dragover', (e) => { e.preventDefault(); screen.classList.add('drag'); });
screen.addEventListener('dragleave', () => screen.classList.remove('drag'));
screen.addEventListener('drop', (e) => {
  e.preventDefault();
  screen.classList.remove('drag');
  const f = e.dataTransfer?.files[0];
  if (f && f.type.startsWith('video')) { loadFile(video, f); started('file', f.name); }
});

$('btn-tab').addEventListener('click', async () => {
  try {
    await loadTab(video);
    tabStart = performance.now() / 1000;
    started('tab', 'Shared tab');
    video.srcObject && (video.srcObject as MediaStream).getVideoTracks()[0]
      .addEventListener('ended', () => { stopStream(video); document.body.dataset.state = 'empty'; });
  } catch {
    showNotice('tab');
  }
});

const play = $('play');
const togglePlay = () => { if (video.paused) video.play(); else video.pause(); };
play.addEventListener('click', togglePlay);
video.addEventListener('play', () => (document.body.dataset.playing = 'true'));
video.addEventListener('pause', () => (document.body.dataset.playing = 'false'));
video.addEventListener('seeking', resetMotion);
video.addEventListener('loadedmetadata', () => { $('m-res').textContent = `${video.videoWidth}×${video.videoHeight}`; });
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && source && !(e.target instanceof HTMLInputElement) && !(e.target as HTMLElement).isContentEditable) {
    e.preventDefault();
    togglePlay();
  }
});
$<HTMLSelectElement>('speed').addEventListener('change', (e) => {
  video.playbackRate = Number((e.target as HTMLSelectElement).value);
});
$<HTMLInputElement>('scrub').addEventListener('input', (e) => {
  if (video.duration) video.currentTime = (Number((e.target as HTMLInputElement).value) / 1000) * video.duration;
});

for (const k of ['markers', 'ids', 'trail'] as const) {
  $<HTMLInputElement>(`t-${k}`).addEventListener('change', (e) => (opts[k] = (e.target as HTMLInputElement).checked));
}

document.querySelectorAll<HTMLButtonElement>('[data-tab]').forEach((b) =>
  b.addEventListener('click', () => {
    tab = b.dataset.tab!;
    document.querySelectorAll('[data-tab]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    document.querySelectorAll<HTMLElement>('.pane').forEach((p) => (p.hidden = p.id !== `pane-${tab}`));
    panel();
  }),
);
document.querySelectorAll<HTMLButtonElement>('[data-heat]').forEach((b) =>
  b.addEventListener('click', () => {
    const v = b.dataset.heat!;
    heatView = v === 'ball' ? 'ball' : (Number(v) as 0 | 1);
    document.querySelectorAll('[data-heat]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    panel();
  }),
);
document.querySelectorAll<HTMLButtonElement>('[data-net]').forEach((b) =>
  b.addEventListener('click', () => {
    netView = Number(b.dataset.net) as 0 | 1;
    document.querySelectorAll('[data-net]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    panel();
  }),
);

$('timeline').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-t]');
  if (b && source === 'file') video.currentTime = Math.max(0, Number(b.dataset.t) - 2);
});

// Calibration: click four reference points on the footage.
const refSelect = $<HTMLSelectElement>('calib-ref');
refSelect.innerHTML = Object.entries(REFERENCES).map(([k, r]) => `<option value="${k}">${r.label}</option>`).join('');
const CORNER = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
function calibHint(): void {
  const h = $('calib-hint');
  h.hidden = !calib;
  document.body.dataset.calib = calib ? 'on' : 'off';
  if (calib) h.textContent = `Click point ${calib.length + 1} of 4 — ${CORNER[calib.length]} of “${REFERENCES[calibRef].label}”. Esc cancels.`;
}
$('btn-calib').addEventListener('click', () => {
  calibRef = refSelect.value;
  calib = [];
  video.pause();
  calibHint();
});
overlay.addEventListener('click', (e) => {
  if (!calib || calib.length >= 4 || !video.videoWidth) return;
  const r = overlay.getBoundingClientRect();
  const v = fitView(overlay, video.videoWidth, video.videoHeight);
  calib.push({ x: (e.clientX - r.left - v.ox) / v.s, y: (e.clientY - r.top - v.oy) / v.s });
  if (calib.length === 4) {
    H = homography(calib, REFERENCES[calibRef].points);
    showNotice(H ? null : 'degenerate');
    stats.forgetPositions();
    const pts = calib;
    setTimeout(() => { if (calib === pts) { calib = null; calibHint(); } }, 900);
    $('calib-hint').hidden = true;
  } else calibHint();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && calib) { calib = null; calibHint(); } });

$('btn-reset').addEventListener('click', () => {
  stats = new MatchStats();
  timeline = [];
  resetMotion();
  panel();
});

function download(name: string, body: string, type: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([body], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$('btn-json').addEventListener('click', () => {
  const names = [$('name-a').textContent, $('name-b').textContent];
  download('football-vision.json', JSON.stringify({
    clip: $('m-clip').textContent, teams: names, calibrated: !!H,
    possession: stats.possessionShare(), passes: stats.passes, turnoversWon: stats.turnoversWon,
    ballVisibleShare: stats.frames ? stats.ballFrames / stats.frames : 0,
    players: [...stats.players.values()].map(({ id, team, metres, touches, n }) => ({ id, team, metres: Math.round(metres), touches, samples: n })),
    events: [...timeline].reverse(),
  }, null, 2), 'application/json');
});
$('btn-csv').addEventListener('click', () => {
  const rows = [...timeline].reverse().map((e) => [e.t.toFixed(2), e.type, e.team, e.from, e.to].join(','));
  download('football-vision-events.csv', ['time_s,type,team,from_id,to_id', ...rows].join('\n'), 'text/csv');
});

// ---------- boot ----------

$('pitch-heat').innerHTML = pitchSvg();
$('net').innerHTML = pitchSvg();
document.body.dataset.state = 'empty';

const loadModel = () => detector.load(import.meta.env.BASE_URL, (f) => ($('loadbar').style.transform = `scaleX(${f})`));
retry(loadModel, MODEL_ATTEMPTS, 1500, (n, err) => console.warn(`Model load attempt ${n} failed`, err))
  .then((meta) => {
    ready = true;
    document.body.dataset.model = 'ready';
    $('m-model').textContent = `${meta.label ?? meta.file} · ${meta.input}px`;
    $('m-engine').textContent = detector.backend === 'webgpu' ? 'WebGPU' : 'WebAssembly';
  })
  .catch((err) => {
    console.error(err);
    modelFailed = true;
    $('m-model').textContent = 'Failed';
    showNotice('model');
  });

requestAnimationFrame(frame);
setInterval(panel, 250);
panel();
