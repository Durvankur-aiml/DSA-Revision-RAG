import { Play } from 'lucide-react';
import { formatPercent, formatTimestamp } from '../utils/format';

/**
 * One retrieved source: video title, timestamp, relevance, and a
 * YouTube deep link provided by the backend (used as-is — the
 * frontend never reconstructs timestamps or URLs, and never re-sorts
 * sources; the backend's order is the relevance order).
 *
 * Relevance is communicated visually (a proportionally filled bar)
 * rather than as a raw internal float.
 */
export default function SourceCard({ source, index }) {
  if (!source) return null;

  const { title, timestamp, url, score } = source;
  const watchLabel = `Watch at ${formatTimestamp(timestamp)}`;

  return (
    <article className="source-card">
      <div className="source-rank" aria-hidden="true">
        {String(index + 1).padStart(2, '0')}
      </div>

      <div className="source-body">
        <h3 className="source-title">{title}</h3>

        <div
          className="source-relevance"
          role="img"
          aria-label={`Relevance ${formatPercent(score)}`}
          title={`Relevance: ${formatPercent(score)}`}
        >
          <span
            className="source-relevance-fill"
            style={{ width: `${Math.round((score ?? 0) * 100)}%` }}
          />
        </div>

        <div className="source-meta">
          <span className="source-meta-item">
            <span className="meta-value">{formatTimestamp(timestamp)}</span>
          </span>
        </div>

        {url ? (
          <a
            className="watch-button"
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`${watchLabel} on YouTube (opens in a new tab)`}
          >
            <Play size={12} aria-hidden="true" />
            {watchLabel}
          </a>
        ) : (
          <span className="watch-unavailable">No link available</span>
        )}
      </div>
    </article>
  );
}
