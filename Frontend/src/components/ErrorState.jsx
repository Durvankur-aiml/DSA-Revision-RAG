/**
 * Error display for failed /ask requests. Shows the user-safe message
 * produced by the API client (never stack traces or internals) and a
 * manual retry button — the user, not the app, controls retries so no
 * Gemini quota is consumed accidentally.
 */
export default function ErrorState({ message, onRetry }) {
  return (
    <section className="error-state" role="alert" aria-live="assertive">
      <div className="error-icon" aria-hidden="true">
        !
      </div>
      <div className="error-body">
        <h2 className="error-title">Something went wrong</h2>
        <p className="error-message">{message}</p>
        {onRetry && (
          <button type="button" className="retry-button" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    </section>
  );
}
