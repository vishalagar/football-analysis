/** Axis-aligned box in source-video pixels (top-left origin). */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type Kind = 'player' | 'goalkeeper' | 'referee' | 'ball';

export interface Detection extends Box {
  kind: Kind;
  score: number;
}

/** -1 = not yet classified, 0 / 1 = the two sides, 2 = officials and outliers. */
export type Team = -1 | 0 | 1 | 2;

export interface Point {
  x: number;
  y: number;
}

export type Rgb = [number, number, number];

export interface MatchEvent {
  t: number;
  type: 'pass' | 'turnover';
  team: 0 | 1;
  from: number;
  to: number;
}
