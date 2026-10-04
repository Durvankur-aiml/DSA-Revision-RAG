import { useCallback, useRef, useState } from 'react';
import { MotionConfig } from 'framer-motion';
import { askQuestion } from './api';
import Header from './components/Header';
import Footer from './components/layout/Footer';
import EmptyState from './components/EmptyState';
import QuestionInput from './components/QuestionInput';
import Conversation from './components/Conversation';
import LoadingState from './components/LoadingState';
import ErrorState from './components/ErrorState';

/**
 * ALGOFORGE — Navigate the world of algorithms.
 *
 * Shell + chat state. (The internal project codename must never
 * appear in user-facing UI.)
 *
 * State model:
 *   messages[]   full exchange history:
 *                  { id, role: "user", question }
 *                  { id, role: "assistant", answer, sources }
 *                  { id, role: "assistant", error, question, failed }
 *   isLoading    exactly one /ask request may be in flight
 *   lastError    connection-level error before any transcript exists
 *
 * Each question is an independent /ask call; history is display-only
 * (the backend contract takes a single question, no conversation).
 * A failed request stays in the transcript as an inline error on its
 * own assistant message, with a Retry action (manual, quota-safe).
 *
 * Motion: MotionConfig reducedMotion="user" disables animation for
 * users with prefers-reduced-motion.
 */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastError, setLastError] = useState(null);

  // Monotonic id counter survives HMR re-evaluation (module-level
  // counters reset on hot reload and duplicate keys).
  const nextMessageIdRef = useRef(1);
  const nextMessageId = () => nextMessageIdRef.current++;
  const isLoadingRef = useRef(false);

  const handleAsk = useCallback(async (rawQuestion) => {
    const trimmed = (rawQuestion ?? '').trim();
    if (!trimmed) return; // composer already blocks; defense in depth
    if (isLoadingRef.current) return; // no duplicate submissions

    isLoadingRef.current = true;
    setIsLoading(true);
    setLastError(null);

    setMessages((prev) => [
      ...prev,
      { id: nextMessageId(), role: 'user', question: trimmed },
    ]);

    try {
      const response = await askQuestion(trimmed);
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: 'assistant',
          answer: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (error) {
      setLastError(error.message);
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: 'assistant',
          error: error.message,
          question: trimmed,
          failed: true,
        },
      ]);
    } finally {
      isLoadingRef.current = false;
      setIsLoading(false);
    }
  }, []);

  function handleRetry(question) {
    if (!question || isLoadingRef.current) return;
    // Remove the failed assistant message, then re-ask. If this was
    // the connection-level error, clear it too.
    setLastError(null);
    setMessages((prev) => {
      const index = prev.findIndex(
        (m) => m.failed && m.question === question
      );
      if (index === -1) return prev;
      const copy = [...prev];
      copy.splice(index, 1);
      return copy;
    });
    handleAsk(question);
  }

  const hasTranscript = messages.length > 0;

  return (
    <MotionConfig reducedMotion="user">
      <div className="app">
        <Header />

        <main className="main">
          {!hasTranscript && !isLoading ? (
            <>
              {lastError && (
                <ErrorState message={lastError} onRetry={undefined} />
              )}
              <EmptyState onExampleSelect={handleAsk} />
            </>
          ) : (
            <>
              <Conversation
                messages={messages}
                isLoading={isLoading}
                onRetry={handleRetry}
              />
              {isLoading && <LoadingState />}
            </>
          )}

          <QuestionInput onSubmit={handleAsk} disabled={isLoading} />
        </main>

        <Footer />
      </div>
    </MotionConfig>
  );
}
