/**
 * Runs `fn` up to `attempts` times, waiting `delayMs × attempt` between tries.
 * A dropped connection while fetching the runtime or model should not leave
 * the app stuck on "Loading…" for good.
 */
export async function retry<T>(
  fn: () => Promise<T>,
  attempts: number,
  delayMs: number,
  onFail?: (attempt: number, err: unknown) => void,
): Promise<T> {
  let last: unknown;
  for (let i = 1; i <= attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      last = err;
      onFail?.(i, err);
      if (i < attempts) await new Promise((r) => setTimeout(r, delayMs * i));
    }
  }
  throw last;
}
