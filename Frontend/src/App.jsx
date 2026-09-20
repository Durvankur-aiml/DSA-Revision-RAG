import { useCallback, useRef, useState } from 'react';
import { askQuestion } from './api';
import Header from './components/Header';
import EmptyState from './components/EmptyState';
import QuestionInput from './components/QuestionInput';
import Conversation from './components/Conversation';
import LoadingState from './components/LoadingState';
import ErrorState from './components/ErrorState';

let nextMessageId = 1;

/**
 * Mission Anthropic — Striver A2Z Knowledge Engine.
 *
 * State model:
 *   messages[]   completed question/answer pairs in this session
 *   isLoading    exactly one /ask request may be in flight
 *   lastError    user-safe error shown with a manual retry
 *
 * Each question is an independent /ask call; history is display-only
 * (the backend contract takes a single question, no conversation).
 */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastError, setError] = useState(null);
  // The last question that failed, for the manual "Try again" button.
  const lastQuestionRef = useRef(null);

  const handleAsk = useCallback(
    async (question) => {
      const trimmed = (question ?? '').trim();
      if (!trimmed || isLoading) return; // no duplicate submissions

      setError(null);
      setIsLoading(true);

      setMessages((prev) => [
        ...prev,
        { id: nextMessageId++, role: 'user', question: trimmed },
      ]);
      lastQuestionRef.current = trimmed;

      try {
        const response = await askQuestion(trimmed);
        setMessages((prev) => [
          ...prev,
          {
            id: nextMessageId++,
            role: 'assistant',
            answer: response.answer,
            sources: response.sources,
          },
        ]);
      } catch (error) {
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading]
  );

  function handleRetry() {
    if (lastQuestionRef.current) {
      handleAsk(lastQuestionRef.current);
    }
  }

  return (
    <div className="app">
      <Header />

      <main className="main">
        {messages.length === 0 && !isLoading && !lastError ? (
          <EmptyState onExampleSelect={handleAsk} />
        ) : (
          <Conversation messages={messages} />
        )}

        {isLoading && <LoadingState />}
        {lastError && !isLoading && (
          <ErrorState message={lastError} onRetry={handleRetry} />
        )}

        <QuestionInput onSubmit={handleAsk} disabled={isLoading} />
      </main>

      <footer className="footer">
        <p>
          Answers are grounded in the indexed Striver A2Z course. Sources link
          to the exact video timestamp.
        </p>
      </footer>
    </div>
  );
}
