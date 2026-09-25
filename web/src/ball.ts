import { dist } from './geometry';
import type { Detection, Point } from './types';

export interface TrailPoint extends Point {
  t: number;
  /** True when the point was filled in between two real sightings. */
  interp: boolean;
}

/**
 * Keeps one ball: picks the most plausible detection each step, rejects
 * teleports, and fills short gaps linearly so the trail stays continuous.
 */
export class BallTrack {
  trail: TrailPoint[] = [];
  seen = 0;
  steps = 0;

  constructor(
    private readonly maxGap = 0.5,
    private readonly trailSeconds = 1.6,
    /** Max plausible speed in frame-widths per second. */
    private readonly maxSpeed = 1.5,
    public frameW = 1280,
  ) {}

  get last(): TrailPoint | null {
    return this.trail.at(-1) ?? null;
  }

  /** Current ball position if it was seen recently. */
  current(t: number): Point | null {
    const l = this.last;
    return l && t - l.t <= this.maxGap ? l : null;
  }

  update(dets: Detection[], t: number): Point | null {
    this.steps++;
    const balls = dets.filter((d) => d.kind === 'ball');
    const prev = this.last;
    let pick: Point | null = null;
    let best = -Infinity;
    for (const b of balls) {
      const p = { x: b.x + b.w / 2, y: b.y + b.h / 2 };
      if (prev && t - prev.t <= this.maxGap) {
        const limit = this.maxSpeed * this.frameW * Math.max(t - prev.t, 0.05);
        if (dist(p, prev) > limit) continue;
      }
      const score = b.score - (prev ? dist(p, prev) / this.frameW : 0);
      if (score > best) { best = score; pick = p; }
    }
    if (pick) {
      this.seen++;
      if (prev && t - prev.t > 0.05 && t - prev.t <= this.maxGap) {
        const n = Math.round((t - prev.t) / 0.05);
        for (let i = 1; i < n; i++) {
          const f = i / n;
          this.trail.push({
            t: prev.t + (t - prev.t) * f,
            x: prev.x + (pick.x - prev.x) * f,
            y: prev.y + (pick.y - prev.y) * f,
            interp: true,
          });
        }
      }
      this.trail.push({ ...pick, t, interp: false });
    }
    this.trail = this.trail.filter((p) => t - p.t <= this.trailSeconds && p.t <= t);
    return pick;
  }

  reset(): void {
    this.trail = [];
  }
}
