import type { Rgb, Team } from './types';

/** Turf is excluded before averaging a torso crop, or every jersey turns green. */
export function isGrass([r, g, b]: Rgb): boolean {
  return g > r * 1.08 && g > b * 1.08 && g > 50;
}

/** Mean colour of the non-grass pixels in an RGBA buffer, or null if mostly turf. */
export function jerseyColour(px: Uint8ClampedArray): Rgb | null {
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < px.length; i += 4) {
    const c: Rgb = [px[i], px[i + 1], px[i + 2]];
    if (isGrass(c)) continue;
    r += c[0]; g += c[1]; b += c[2]; n++;
  }
  return n >= px.length / 4 / 5 ? [r / n, g / n, b / n] : null;
}

const d2 = (a: Rgb, b: Rgb) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;

/**
 * Plain k-means with farthest-point initialisation, which is deterministic
 * and splits two strongly contrasting kits reliably.
 */
export function kmeans(points: Rgb[], k: number, iters = 20): Rgb[] {
  if (points.length < k) return [];
  const cents: Rgb[] = [points[0]];
  while (cents.length < k) {
    let far = points[0], best = -1;
    for (const p of points) {
      const m = Math.min(...cents.map((c) => d2(p, c)));
      if (m > best) { best = m; far = p; }
    }
    cents.push(far);
  }
  for (let it = 0; it < iters; it++) {
    const sum = cents.map(() => [0, 0, 0, 0]);
    for (const p of points) {
      const s = sum[nearest(p, cents)];
      s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; s[3]++;
    }
    sum.forEach((s, i) => { if (s[3]) cents[i] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]; });
  }
  return cents;
}

export function nearest(p: Rgb, cents: Rgb[]): number {
  let bi = 0;
  cents.forEach((c, i) => { if (d2(p, c) < d2(p, cents[bi])) bi = i; });
  return bi;
}

/**
 * Collects jersey colours until there are enough to fit two kits, then
 * labels each colour as team 0, team 1 or 2 (officials and outliers).
 * Three clusters are fitted first; the two biggest become the teams.
 */
export class TeamModel {
  centres: Rgb[] = [];
  /** Kit of the officials, when they formed their own cluster. */
  officials: Rgb | null = null;
  private samples: Rgb[] = [];
  private radius = Infinity;

  constructor(private readonly needed = 30) {}

  get ready(): boolean {
    return this.centres.length === 2;
  }

  add(c: Rgb): void {
    if (this.ready) return;
    this.samples.push(c);
    if (this.samples.length >= this.needed) this.fit();
  }

  private fit(): void {
    const n = this.samples.length;
    const k3 = kmeans(this.samples, 3);
    const counts = k3.map(() => 0);
    for (const s of this.samples) counts[nearest(s, k3)]++;
    const [a, b, c] = counts.map((cnt, i) => [cnt, i]).sort((x, y) => y[0] - x[0]).map(([, i]) => k3[i]);
    // The third cluster is the officials only if it is a real share of samples
    // and a genuinely different colour, not a shading split of one kit.
    const apart = Math.sqrt(Math.min(d2(c, a), d2(c, b))) > 0.5 * Math.sqrt(d2(a, b));
    if (counts[k3.indexOf(c)] >= n * 0.04 && apart) {
      this.centres = [a, b];
      this.officials = c;
    } else {
      this.centres = kmeans(this.samples, 2);
      this.officials = null;
    }
    const ds = this.samples
      .filter((s) => this.classifyCluster(s) !== 2)
      .map((s) => Math.sqrt(d2(s, this.centres[nearest(s, this.centres)])))
      .sort((x, y) => x - y);
    this.radius = Math.max(40, (ds[Math.floor(ds.length * 0.8)] ?? 40) * 1.8);
  }

  private classifyCluster(c: Rgb): Team {
    const i = nearest(c, this.centres);
    return this.officials && d2(c, this.officials) < d2(c, this.centres[i]) ? 2 : (i as 0 | 1);
  }

  classify(c: Rgb | null): Team {
    if (!c || !this.ready) return -1;
    const t = this.classifyCluster(c);
    if (t === 2) return 2;
    return Math.sqrt(d2(c, this.centres[t])) > this.radius ? 2 : t;
  }

  reset(): void {
    this.centres = [];
    this.officials = null;
    this.samples = [];
    this.radius = Infinity;
  }
}
