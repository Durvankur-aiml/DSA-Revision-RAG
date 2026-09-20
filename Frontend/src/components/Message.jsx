import Markdown from '../markdown';
import SourceCard from './SourceCard';

/**
 * One question/answer exchange.
 *
 * Props:
 *   message.role      "user" | "assistant"
 *   message.question  the user's question (assistant messages)
 *   message.answer    generated answer text (assistant messages)
 *   message.sources   Source[] (assistant messages)
 *   message.error     user-safe error message (failed assistant messages)
 */
export default function Message({ message }) {
  if (message.role === 'user') {
    return (
      <article className="message message-user" aria-label="Your question">
        <p className="message-question">{message.question}</p>
      </article>
    );
  }

  return (
    <article className="message message-assistant" aria-label="Answer">
      <p className="message-label">ANSWER</p>

      {message.error ? (
        <ErrorStateInline message={message.error} />
      ) : (
        <>
          <Markdown text={message.answer} />

          {message.sources?.length > 0 && (
            <section className="sources-section" aria-label="Sources">
              <p className="message-label">SOURCES</p>
              <div className="sources-grid">
                {message.sources.map((source, index) => (
                  <SourceCard key={index} source={source} index={index} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </article>
  );
}

function ErrorStateInline({ message }) {
  return (
    <div className="error-state error-inline" role="alert">
      <div className="error-icon" aria-hidden="true">
        !
      </div>
      <div className="error-body">
        <p className="error-message">{message}</p>
      </div>
    </div>
  );
}
