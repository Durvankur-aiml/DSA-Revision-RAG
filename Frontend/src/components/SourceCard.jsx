import { formatScore, formatTimestamp } from '../utils/format';

/**
 * One retrieved source: video title, relevance, timestamp, and a
 * YouTube deep link provided by the backend (used as-is — the frontend
 * never reconstructs timestamps or URLs).
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

        <div className="source-meta">
          <span className="source-meta-item">
            <span className="meta-label">RELEVANCE</span>
            <span className="meta-value">{formatScore(score)}</span>
          </span>
          <span className="source-meta-item">
            <span className="meta-label">TIMESTAMP</span>
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
            <span aria-hidden="true">▶</span> {watchLabel.toUpperCase()}
          </a>
        ) : (
          <span className="watch-unavailable">No link available</span>
        )}
      </div>
    </article>
  );
}
