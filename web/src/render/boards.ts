import { PITCH } from '../pitch';
import { GRID, type MatchStats } from '../stats';
import type { Rgb } from '../types';

/**
 * Paints a heat grid as chalk: a blurred density, then per-pixel grain so it
 * reads like rubbed chalk on a tactics board rather than a smooth gradient.
 */
export function drawHeat(canvas: HTMLCanvasElement, grid: Float32Array, rgb: Rgb): void {
  const W = 420, H = Math.round((W * PITCH.width) / PITCH.length);
  if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...grid);
  if (max <= 0) return;
  const cw = W / GRID.cols, ch = H / GRID.rows;
  ctx.filter = `blur(${cw * 0.9}px)`;
  const [r, g, b] = chalkTint(rgb);
  for (let i = 0; i < grid.length; i++) {
    if (!grid[i]) continue;
    const a = Math.pow(grid[i] / max, 0.6);
    ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
    ctx.fillRect((i % GRID.cols) * cw - cw * 0.25, Math.floor(i / GRID.cols) * ch - ch * 0.25, cw * 1.5, ch * 1.5);
  }
  ctx.filter = 'none';
  const img = ctx.getImageData(0, 0, W, H);
  let seed = 7;
  for (let i = 3; i < img.data.length; i += 4) {
    seed = (seed * 16807) % 2147483647;
    img.data[i] *= 0.55 + 0.45 * (seed / 2147483647);
  }
  ctx.putImageData(img, 0, 0);
}

/** Pushes a kit colour towards chalk so dark kits still show on turf. */
export function chalkTint([r, g, b]: Rgb): Rgb {
  const m = 0.45;
  return [r + (236 - r) * m, g + (234 - g) * m, b + (224 - b) * m].map(Math.round) as Rgb;
}

/**
 * Pass network: each player at their average position, sized by how often
 * they were involved, joined by lines as thick as the passes between them.
 */
export function networkSvg(stats: MatchStats, team: 0 | 1, colour: string): string {
  const nodes = [...stats.players.values()].filter((p) => p.team === team && p.n >= 8);
  const byId = new Map(nodes.map((p) => [p.id, { x: p.sum.x / p.n, y: p.sum.y / p.n, p }]));
  const pairs = new Map<string, { a: number; b: number; n: number }>();
  for (const [k, n] of stats.edges) {
    const [a, b] = k.split('>').map(Number);
    if (!byId.has(a) || !byId.has(b)) continue;
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    const e = pairs.get(key) ?? { a, b, n: 0 };
    e.n += n;
    pairs.set(key, e);
  }
  const maxN = Math.max(1, ...[...pairs.values()].map((e) => e.n));
  const lines = [...pairs.values()].map(({ a, b, n }) => {
    const p = byId.get(a)!, q = byId.get(b)!;
    return `<line x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}" stroke-width="${0.3 + (n / maxN) * 1.6}"/>`;
  });
  const top = nodes.sort((a, b) => b.n - a.n).slice(0, 14);
  const maxP = Math.max(1, ...top.map((p) => p.n));
  const dots = top.map((p) => {
    const c = byId.get(p.id)!;
    const r = 1.4 + Math.sqrt(p.n / maxP) * 1.8;
    return `<g class="node"><circle cx="${c.x}" cy="${c.y}" r="${r}" fill="${colour}"/>` +
      `<text x="${c.x}" y="${c.y + 0.9}">${p.id}</text></g>`;
  });
  return `<g class="net">${lines.join('')}</g>${dots.join('')}`;
}
