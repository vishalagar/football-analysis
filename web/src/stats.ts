import { PITCH } from './pitch';
import type { MatchEvent, Point, Team } from './types';

export const GRID = { cols: 21, rows: 14 }; // 5 m cells on a 105 × 68 pitch

export interface PlayerStats {
  id: number;
  team: Team;
  metres: number;
  touches: number;
  sum: Point;
  n: number;
  last: (Point & { t: number }) | null;
}

export interface Sample {
  id: number;
  team: Team;
  /** Position on the pitch in metres. */
  pos: Point;
  /** Metres per image pixel near this player, used when uncalibrated. */
  mpp: number;
  /** Raw foot point in image pixels. */
  px: Point;
}

/** Top sprint speed; anything faster between samples is a tracking glitch. */
const MAX_SPEED = 11;

export class MatchStats {
  possession: [number, number] = [0, 0];
  passes: [number, number] = [0, 0];
  turnoversWon: [number, number] = [0, 0];
  players = new Map<number, PlayerStats>();
  heat: [Float32Array, Float32Array] = [newGrid(), newGrid()];
  ballHeat = newGrid();
  edges = new Map<string, number>();
  frames = 0;
  ballFrames = 0;

  /** dt: seconds since the previous analysed frame. */
  step(
    t: number, dt: number, samples: Sample[], ball: Point | null,
    team: 0 | 1 | null, calibrated: boolean,
  ): void {
    this.frames++;
    if (ball) {
      this.ballFrames++;
      bump(this.ballHeat, ball);
    }
    if (team !== null && dt < 1) this.possession[team] += dt;
    for (const s of samples) {
      let p = this.players.get(s.id);
      if (!p) {
        p = { id: s.id, team: s.team, metres: 0, touches: 0, sum: { x: 0, y: 0 }, n: 0, last: null };
        this.players.set(s.id, p);
      }
      p.team = s.team;
      if (p.last && t > p.last.t && t - p.last.t <= 1.5) {
        const m = calibrated
          ? Math.hypot(s.pos.x - p.last.x, s.pos.y - p.last.y)
          : Math.hypot(s.px.x - p.last.x, s.px.y - p.last.y) * s.mpp;
        if (m / (t - p.last.t) <= MAX_SPEED) p.metres += m;
      }
      p.last = calibrated ? { ...s.pos, t } : { ...s.px, t };
      p.sum.x += s.pos.x;
      p.sum.y += s.pos.y;
      p.n++;
      if (s.team === 0 || s.team === 1) bump(this.heat[s.team], s.pos);
    }
  }

  event(ev: MatchEvent): void {
    const to = this.players.get(ev.to);
    if (to) to.touches++;
    if (ev.type === 'pass') {
      this.passes[ev.team]++;
      const k = `${ev.from}>${ev.to}`;
      this.edges.set(k, (this.edges.get(k) ?? 0) + 1);
    } else {
      this.turnoversWon[ev.team]++;
    }
  }

  possessionShare(): [number, number] {
    const [a, b] = this.possession;
    if (a + b === 0) return [50, 50];
    const pa = Math.round((a / (a + b)) * 100);
    return [pa, 100 - pa];
  }

  teamMetres(team: 0 | 1): number {
    let m = 0;
    for (const p of this.players.values()) if (p.team === team) m += p.metres;
    return m;
  }

  /** Clears positional state after a camera cut but keeps the totals. */
  forgetPositions(): void {
    for (const p of this.players.values()) p.last = null;
  }
}

function newGrid(): Float32Array {
  return new Float32Array(GRID.cols * GRID.rows);
}

function bump(g: Float32Array, p: Point): void {
  const c = Math.floor((p.x / PITCH.length) * GRID.cols);
  const r = Math.floor((p.y / PITCH.width) * GRID.rows);
  if (c >= 0 && c < GRID.cols && r >= 0 && r < GRID.rows) g[r * GRID.cols + c]++;
}
