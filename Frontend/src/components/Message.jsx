import { motion } from 'framer-motion';
import { AlertCircle, RotateCcw } from 'lucide-react';
import Markdown from '../markdown';
import SourceCard from './SourceCard';

/**
 * One exchange in the conversation.
 *
 * message.role "user":      the question bubble
 * message.role "assistant": answer (Markdown + sources) — or, when
 * message.failed, an inline error card with a Retry action. Failed
 * requests stay in the transcript (with the question that failed)
 * instead of vanishing into a global error box.
 */
export default function Message({ message, onRetry }) {
  if (message.role === 'user') {
    return (
      <motion.article
        className="message message-user"
        aria-label="Your question"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22, ease: 'easeOut' }}
      >
        <p className="message-question">{message.question}</p>
      </motion.article>
    );
  }

  return (
    <motion.article
      className="message message-assistant"
      aria-label={message.failed ? 'Answer failed' : 'Answer'}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.26, ease: 'easeOut' }}
    >
      {message.failed ? (
        <div className="error-inline" role="alert">
          <AlertCircle size={18} aria-hidden="true" className="error-inline-icon" />
          <div className="error-inline-body">
            <p className="error-inline-title">Answer failed</p>
            <p className="error-inline-message">{message.error}</p>
            {onRetry && (
              <button
                type="button"
                className="retry-inline"
                onClick={() => onRetry(message.question)}
              >
                <RotateCcw size={13} aria-hidden="true" />
                Retry
              </button>
            )}
          </div>
        </div>
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
    </motion.article>
  );
}
