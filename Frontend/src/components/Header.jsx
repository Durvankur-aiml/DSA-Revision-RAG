import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

/**
 * Product header: brand on the left, live API status on the right.
 * The status reflects only /health reachability — it never claims
 * anything about Gemini quota availability.
 */
export default function Header() {
  const [online, setOnline] = useState(null); // null = checking

  useEffect(() => {
    let cancelled = false;
    checkHealth().then((ok) => {
      if (!cancelled) setOnline(ok);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">◈</span>
        <span className="brand-text">
          <span className="brand-name">MISSION ANTHROPIC</span>
          <span className="brand-subtitle">STRIVER A2Z KNOWLEDGE ENGINE</span>
        </span>
      </div>

      <div
        className={`status ${online === true ? 'status-online' : ''} ${
          online === false ? 'status-offline' : ''
        }`}
        role="status"
        aria-live="polite"
      >
        <span className="status-dot" aria-hidden="true" />
        {online === null && 'CHECKING API…'}
        {online === true && 'API ONLINE'}
        {online === false && 'API OFFLINE'}
      </div>
    </header>
  );
}
