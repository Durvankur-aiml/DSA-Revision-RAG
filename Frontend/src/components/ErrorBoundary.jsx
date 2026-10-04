import { Component } from 'react';

/**
 * Top-level error boundary.
 *
 * Catches render-time errors anywhere below it (e.g. an unexpected
 * markdown parse edge case) and shows a recoverable full-page error
 * card instead of a white screen. Reuses the existing error-state
 * styling; if the stylesheet itself fails to load, the inline fallback
 * below still keeps the message readable.
 *
 * Error details are logged to the console only — never rendered —
 * so internals never reach the UI.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('UI render error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'grid',
          placeItems: 'center',
          padding: '24px',
        }}
      >
        <div
          className="error-state"
          style={{ maxWidth: '440px', width: '100%' }}
          role="alert"
        >
          <div className="error-icon" aria-hidden="true">
            !
          </div>
          <div className="error-body">
            <h2 className="error-title">Something went wrong</h2>
            <p className="error-message">
              The interface hit an unexpected error. Reloading usually
              resolves it; your question history for this session is kept.
            </p>
            <button
              type="button"
              className="retry-button"
              onClick={this.handleReload}
            >
              Reload ALGOFORGE
            </button>
          </div>
        </div>
      </div>
    );
  }
}
