import { describe, expect, it } from 'vitest';
import { Possession, type Player } from '../src/events';

const pl = (id: number, team: 0 | 1, x: number): Player => ({ id, team, x, y: 0, w: 20, h: 60 });
const feet = (p: Player) => ({ x: p.x + p.w / 2, y: p.y + p.h });

function run(poss: Possession, seq: [Player, number][], players: Player[]) {
  const out = [];
  let t = 0;
  for (const [p, steps] of seq) {
    for (let i = 0; i < steps; i++, t += 0.1) {
      const ev = poss.update(t, feet(p), players);
      if (ev) out.push(ev);
    }
    // ball in flight between players
    for (let i = 0; i < 3; i++, t += 0.1) poss.update(t, { x: 9999, y: 0 }, players);
  }
  return out;
}

describe('Possession', () => {
  const a1 = pl(1, 0, 0), a2 = pl(2, 0, 300), b1 = pl(3, 1, 600);
  const all = [a1, a2, b1];

  it('counts a pass between team-mates', () => {
    const evs = run(new Possession(), [[a1, 4], [a2, 4]], all);
    expect(evs).toEqual([expect.objectContaining({ type: 'pass', team: 0, from: 1, to: 2 })]);
  });

  it('counts a turnover when the other side takes it', () => {
    const evs = run(new Possession(), [[a1, 4], [b1, 4]], all);
    expect(evs).toEqual([expect.objectContaining({ type: 'turnover', team: 1, from: 1, to: 3 })]);
  });

  it('ignores a ball that only brushes past a player', () => {
    const evs = run(new Possession({ confirm: 0.25 }), [[a1, 4], [a2, 1], [a1, 4]], all);
    expect(evs).toHaveLength(0);
  });

  it('credits possession to the last team in control', () => {
    const p = new Possession();
    run(p, [[a1, 4]], all);
    expect(p.team(1)).toBe(0);
    expect(p.team(100)).toBe(null);
  });
});
