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
  /** Seconds a dropped player's id is kept for them to come back. */
  reviveSec: number;
  /** Largest kit colour distance (RGB) still treated as the same player. */
  reviveColour: number;
}

const DEFAULTS: TrackerOptions = {
  iouMin: 0.2, centreGate: 0.8, maxMissed: 12, minHits: 3, reviveSec: 4, reviveColour: 60,
};

/** Samples the kit colour inside a box, if the caller can see the frame. */
export type ColourOf = (b: Box) => Rgb | null;

/** A confirmed player who dropped out of view, kept briefly for re-identification. */
interface Lost extends Box {
  id: number;
  kind: Kind;
  colour: Rgb | null;
  vx: number;
  vy: number;
  lastT: number;
}

/** How far (in body heights) a lost player may have moved after dt seconds. */
const reachAfter = (dt: number) => Math.min(6, 1 + 3 * dt);
const rgbDist = (a: Rgb, b: Rgb) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * SORT-style tracker without a Kalman filter: constant-velocity prediction,
 * greedy IoU matching, then a centre-distance pass for fast movers.
 *
 * Ids are handed out only when a track is confirmed, so flickering false
 * detections don't burn through numbers. A confirmed player who disappears
 * (occlusion, leaving the frame briefly) gets their old id back if they
 * reappear near where they were heading, in the same kit, within reviveSec.
 */
export class Tracker {
  tracks: Track[] = [];
  private lost: Lost[] = [];
  private nextId = 1;
  private readonly opt: TrackerOptions;

  constructor(opt: Partial<TrackerOptions> = {}) {
    this.opt = { ...DEFAULTS, ...opt };
  }

  reset(): void {
    this.tracks = [];
    this.lost = [];
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

  update(dets: Detection[], t: number, colourOf?: ColourOf): Track[] {
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
        ...box(d), id: 0, kind: d.kind, team: -1, colour: null,
        hits: 1, missed: 0, vx: 0, vy: 0, lastT: t,
      });
    }
    this.retire(t);
    for (const tr of this.tracks) if (!tr.id && tr.hits >= this.opt.minHits) this.confirm(tr, t, colourOf);
    return this.confirmed;
  }

  /** Moves dropped tracks to the lost list and forgets players gone too long. */
  private retire(t: number): void {
    const keep: Track[] = [];
    for (const tr of this.tracks) {
      if (tr.missed <= this.opt.maxMissed) keep.push(tr);
      else if (tr.id) this.lost.push({ ...box(tr), id: tr.id, kind: tr.kind, colour: tr.colour, vx: tr.vx, vy: tr.vy, lastT: tr.lastT });
    }
    this.tracks = keep;
    this.lost = this.lost.filter((l) => t - l.lastT <= this.opt.reviveSec);
  }

  /** Gives a newly confirmed track a lost player's id when it fits, else a fresh one. */
  private confirm(tr: Track, t: number, colourOf?: ColourOf): void {
    const colour = colourOf?.(tr) ?? null;
    let best = -1;
    let bestCost = Infinity;
    this.lost.forEach((l, i) => {
      if ((l.kind === 'referee') !== (tr.kind === 'referee')) return;
      const dt = t - l.lastT;
      const k = Math.min(0.5, dt);
      const h = Math.max(l.h, tr.h);
      const d = dist(centre({ ...l, x: l.x + l.vx * k, y: l.y + l.vy * k }), centre(tr)) / h;
      if (d > reachAfter(dt)) return;
      const c = colour && l.colour ? rgbDist(colour, l.colour) : null;
      if (c !== null && c > this.opt.reviveColour) return;
      const cost = d + (c ?? this.opt.reviveColour / 2) / this.opt.reviveColour;
      if (cost < bestCost) { bestCost = cost; best = i; }
    });
    if (best < 0) {
      tr.id = this.nextId++;
      tr.colour = colour;
      return;
    }
    const [l] = this.lost.splice(best, 1);
    tr.id = l.id;
    tr.colour = l.colour ?? colour;
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
