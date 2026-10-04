import { AlertCircle, RotateCcw } from 'lucide-react';

/**
 * Connection-level error card, shown when a request fails before any
 * conversation content exists (the transcript is still empty). Once
 * the conversation has content, failures render inline on the failed
 * message instead (see Message.jsx) — including the Retry action.
 *
 * Displays the user-safe message produced by the API client (never
 * stack traces or internals). Retries are manual by design so no
 * Gemini quota is consumed accidentally.
 */
export default function ErrorState({ message, onRetry }) {
  return (
    <section className="error-state" role="alert" aria-live="assertive">
      <div className="error-icon" aria-hidden="true">
        <AlertCircle size={15} strokeWidth={2.4} />
      </div>
      <div className="error-body">
        <h2 className="error-title">Something went wrong</h2>
        <p className="error-message">{message}</p>
        {onRetry && (
          <button type="button" className="retry-button" onClick={onRetry}>
            <RotateCcw size={13} aria-hidden="true" />
            Try again
          </button>
        )}
      </div>
    </section>
  );
}
