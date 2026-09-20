/**
 * Format seconds as a human-readable video timestamp.
 *
 *   86   -> "01:26"
 *   125  -> "02:05"
 *   3661 -> "01:01:01"
 */
export function formatTimestamp(totalSeconds) {
  const seconds = Number.isFinite(totalSeconds)
    ? Math.max(0, Math.floor(totalSeconds))
    : 0;

  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;

  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');

  return h > 0 ? `${String(h).padStart(2, '0')}:${mm}:${ss}` : `${mm}:${ss}`;
}

/**
 * Format a relevance score for display: 1 -> "1.00", 0.7285 -> "0.73".
 */
export function formatScore(score) {
  const value = Number.isFinite(score) ? Math.min(1, Math.max(0, score)) : 0;
  return value.toFixed(2);
}
