import type { TrailPoint } from '../ball';
import type { Track } from '../tracker';
import type { Point } from '../types';

export interface OverlayOptions {
  markers: boolean;
  ids: boolean;
  trail: boolean;
}

export interface View {
  /** Scale and offset from video pixels to canvas CSS pixels (object-fit: contain). */
  s: number;
  ox: number;
  oy: number;
}

const BALL = '#F2C230';
const CHALK = '#ECEAE0';

export function fitView(canvas: HTMLCanvasElement, vw: number, vh: number): View {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const s = Math.min(r.width / vw, r.height / vh) || 1;
  return { s, ox: (r.width - vw * s) / 2, oy: (r.height - vh * s) / 2 };
}

export function drawOverlay(
  canvas: HTMLCanvasElement,
  v: View,
  tracks: Track[],
  teamColour: (t: Track) => string,
  trail: TrailPoint[],
  t: number,
  holderId: number | null,
  calib: Point[],
  opt: OverlayOptions,
): void {
  const ctx = canvas.getContext('2d')!;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const X = (x: number) => v.ox + x * v.s;
  const Y = (y: number) => v.oy + y * v.s;

  if (opt.markers || opt.ids) {
    for (const tr of tracks) {
      const x = X(tr.x), y = Y(tr.y), w = tr.w * v.s, h = tr.h * v.s;
      const col = teamColour(tr);
      if (opt.markers) brackets(ctx, x, y, w, h, col);
      if (opt.ids) tag(ctx, String(tr.id), x + w / 2, y - 4, col);
      if (tr.id === holderId) {
        ctx.fillStyle = BALL;
        ctx.beginPath();
        const cx = x + w / 2, ty = y + h + 5;
        ctx.moveTo(cx, ty); ctx.lineTo(cx - 5, ty + 7); ctx.lineTo(cx + 5, ty + 7);
        ctx.fill();
      }
    }
  }

  if (opt.trail && trail.length) {
    ctx.lineCap = ctx.lineJoin = 'round';
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      const age = Math.min(1, (t - b.t) / 1.6);
      ctx.strokeStyle = `rgba(242,194,48,${(1 - age) * (b.interp ? 0.45 : 0.9)})`;
      ctx.lineWidth = 2.5 * (1 - age) + 0.5;
      ctx.setLineDash(b.interp ? [3, 4] : []);
      ctx.beginPath();
      ctx.moveTo(X(a.x), Y(a.y));
      ctx.lineTo(X(b.x), Y(b.y));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const last = trail[trail.length - 1];
    if (t - last.t < 0.4) {
      ctx.strokeStyle = BALL;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(X(last.x), Y(last.y), 9, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  if (calib.length) {
    ctx.strokeStyle = BALL;
    ctx.fillStyle = BALL;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    calib.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    if (calib.length === 4) ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
    calib.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '600 12px "IBM Plex Mono", monospace';
      ctx.fillText(String(i + 1), X(p.x) + 8, Y(p.y) - 8);
    });
  }
}

/** Corner brackets rather than a full box keep the footage readable. */
function brackets(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, col: string) {
  const k = Math.max(4, Math.min(w, h) * 0.28);
  ctx.strokeStyle = col;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x + w, y + h, -1, -1], [x, y + h, 1, -1]]) {
    ctx.moveTo(cx + dx * k, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * k);
  }
  ctx.stroke();
}

/** Squad-number style tag above the player's head. */
function tag(ctx: CanvasRenderingContext2D, text: string, cx: number, bottom: number, col: string) {
  ctx.font = '600 10px "IBM Plex Mono", monospace';
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = col;
  ctx.fillRect(cx - w / 2, bottom - 14, w, 14);
  ctx.fillStyle = readableOn(col);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, cx, bottom - 6.5);
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';
}

export function readableOn(css: string): string {
  const m = css.match(/\d+(\.\d+)?/g);
  if (!m) return '#0F1411';
  const [r, g, b] = m.map(Number);
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? '#0F1411' : CHALK;
}
