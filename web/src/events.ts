import { dist, foot } from './geometry';
import type { Box, MatchEvent, Point, Team } from './types';

export interface Player extends Box {
  id: number;
  team: Team;
}

interface Holder {
  id: number;
  team: Team;
  t: number;
}

export interface PossessionOptions {
  /** Ball-to-feet distance that counts as control, as a share of player height. */
  reach: number;
  /** Seconds a player must stay closest to the ball before it is theirs. */
  confirm: number;
  /** After this long without a new holder, the last touch no longer links to the next. */
  memory: number;
}

const DEFAULTS: PossessionOptions = { reach: 0.6, confirm: 0.12, memory: 6 };

/**
 * Decides who controls the ball and emits a pass when control moves between
 * team-mates, or a turnover when it moves to the other side.
 */
export class Possession {
  holder: Holder | null = null;
  last: Holder | null = null;
  events: MatchEvent[] = [];
  private cand: { id: number; since: number } | null = null;
  private readonly opt: PossessionOptions;

  constructor(opt: Partial<PossessionOptions> = {}) {
    this.opt = { ...DEFAULTS, ...opt };
  }

  /** Team credited with the ball: the current holder, else the last recent one. */
  team(t: number): 0 | 1 | null {
    const h = this.holder ?? this.last;
    if (!h || t - h.t > this.opt.memory) return null;
    return h.team === 0 || h.team === 1 ? h.team : null;
  }

  update(t: number, ball: Point | null, players: Player[]): MatchEvent | null {
    if (!ball) {
      this.cand = null;
      if (this.holder && t - this.holder.t > 0.8) this.holder = null;
      return null;
    }
    let near: Player | null = null;
    let best = Infinity;
    for (const p of players) {
      const d = dist(ball, foot(p)) / p.h;
      if (d < this.opt.reach && d < best) { best = d; near = p; }
    }
    if (!near) {
      this.cand = null;
      if (this.holder && t - this.holder.t > 0.3) this.holder = null;
      return null;
    }
    if (this.holder?.id === near.id) {
      this.holder.t = t;
      this.holder.team = near.team;
      this.last = this.holder;
      return null;
    }
    if (this.cand?.id !== near.id) this.cand = { id: near.id, since: t };
    if (t - this.cand.since < this.opt.confirm) return null;
    return this.take(near, t);
  }

  private take(p: Player, t: number): MatchEvent | null {
    const prev = this.last;
    this.holder = { id: p.id, team: p.team, t };
    this.last = this.holder;
    this.cand = null;
    if (!prev || prev.id === p.id || t - prev.t > this.opt.memory) return null;
    if ((prev.team !== 0 && prev.team !== 1) || (p.team !== 0 && p.team !== 1)) return null;
    const ev: MatchEvent = {
      t,
      type: prev.team === p.team ? 'pass' : 'turnover',
      team: p.team,
      from: prev.id,
      to: p.id,
    };
    this.events.push(ev);
    return ev;
  }

  reset(): void {
    this.holder = this.last = this.cand = null;
  }
}
