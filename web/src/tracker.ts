import { centre, dist, iou } from './geometry';
import type { Box, Detection, Kind, Rgb, Team } from './types';

export interface Track extends Box {
  id: number;
  kind: Kind;
  team: Team;
  /** Running jersey colour estimate, filled in by the team classifier. */
  colour: Rgb | null;
  hits: number;
  missed: number;
  vx: number;
  vy: number;
  lastT: number;
}

export interface TrackerOptions {
  /** Minimum IoU to accept a match on the first pass. */
  iouMin: number;
  /** Second pass: centre distance allowed, as a multiple of box height. */
  centreGate: number;
  /** Detection steps a track may go unmatched before it is dropped. */
  maxMissed: number;
  /** Hits needed before a track is shown and counted. */
  minHits: number;
}

const DEFAULTS: TrackerOptions = { iouMin: 0.2, centreGate: 0.8, maxMissed: 12, minHits: 3 };

/**
 * SORT-style tracker without a Kalman filter: constant-velocity prediction,
 * greedy IoU matching, then a centre-distance pass for fast movers.
 */
export class Tracker {
  tracks: Track[] = [];
  private nextId = 1;
  private readonly opt: TrackerOptions;

  constructor(opt: Partial<TrackerOptions> = {}) {
    this.opt = { ...DEFAULTS, ...opt };
  }

  reset(): void {
    this.tracks = [];
  }

  get confirmed(): Track[] {
    return this.tracks.filter((t) => t.hits >= this.opt.minHits && t.missed === 0);
  }

  /** Positions extrapolated to time t, for smooth drawing between detections. */
  predicted(t: number): Track[] {
    return this.tracks
      .filter((tr) => tr.hits >= this.opt.minHits && tr.missed <= 2)
      .map((tr) => {
        const dt = Math.min(0.5, Math.max(0, t - tr.lastT));
        return { ...tr, x: tr.x + tr.vx * dt, y: tr.y + tr.vy * dt };
      });
  }

  update(dets: Detection[], t: number): Track[] {
    const people = dets.filter((d) => d.kind !== 'ball');
    const unmatchedT = new Set(this.tracks.map((_, i) => i));
    const unmatchedD = new Set(people.map((_, i) => i));
    const pred = this.tracks.map((tr) => {
      const dt = Math.min(0.5, Math.max(0, t - tr.lastT));
      return { x: tr.x + tr.vx * dt, y: tr.y + tr.vy * dt, w: tr.w, h: tr.h };
    });

    const pairs: [number, number, number][] = [];
    this.tracks.forEach((_, ti) =>
      people.forEach((d, di) => {
        const s = iou(pred[ti], d);
        if (s >= this.opt.iouMin) pairs.push([s, ti, di]);
      }),
    );
    this.assign(pairs, unmatchedT, unmatchedD, people, t);

    const far: [number, number, number][] = [];
    for (const ti of unmatchedT)
      for (const di of unmatchedD) {
        const d = dist(centre(pred[ti]), centre(people[di]));
        const gate = this.opt.centreGate * Math.max(pred[ti].h, people[di].h);
        if (d < gate) far.push([-d, ti, di]);
      }
    this.assign(far, unmatchedT, unmatchedD, people, t);

    for (const ti of unmatchedT) this.tracks[ti].missed++;
    for (const di of unmatchedD) {
      const d = people[di];
      this.tracks.push({
        ...box(d), id: this.nextId++, kind: d.kind, team: -1, colour: null,
        hits: 1, missed: 0, vx: 0, vy: 0, lastT: t,
      });
    }
    this.tracks = this.tracks.filter((tr) => tr.missed <= this.opt.maxMissed);
    return this.confirmed;
  }

  private assign(
    pairs: [number, number, number][],
    freeT: Set<number>,
    freeD: Set<number>,
    people: Detection[],
    t: number,
  ): void {
    pairs.sort((a, b) => b[0] - a[0]);
    for (const [, ti, di] of pairs) {
      if (!freeT.has(ti) || !freeD.has(di)) continue;
      freeT.delete(ti);
      freeD.delete(di);
      const tr = this.tracks[ti];
      const d = people[di];
      const dt = t - tr.lastT;
      if (dt > 1e-3) {
        const a = 0.5; // velocity smoothing
        tr.vx = a * ((d.x - tr.x) / dt) + (1 - a) * tr.vx;
        tr.vy = a * ((d.y - tr.y) / dt) + (1 - a) * tr.vy;
      }
      Object.assign(tr, box(d));
      tr.kind = d.kind;
      tr.hits++;
      tr.missed = 0;
      tr.lastT = t;
    }
  }
}

const box = (b: Box): Box => ({ x: b.x, y: b.y, w: b.w, h: b.h });
