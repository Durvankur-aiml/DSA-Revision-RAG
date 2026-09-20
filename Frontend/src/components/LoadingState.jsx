import { useEffect, useState } from 'react';

/**
 * Loading state shown while POST /ask is in flight.
 *
 * This is a UI indicator only: it NEVER fabricates partial answer
 * text or fakes token streaming. The two phases are visual pacing
 * while the single request completes.
 */
export default function LoadingState() {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    // Switch to the "generating" phase after a short delay — retrieval
    // typically completes within the first seconds of the request.
    const timer = setTimeout(() => setPhase(1), 2500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section className="loading-state" aria-live="polite" aria-busy="true">
      <div className="loading-spinner" aria-hidden="true" />

      {phase === 0 ? (
        <>
          <h2 className="loading-title">SEARCHING KNOWLEDGE BASE</h2>
          <p className="loading-subtitle">
            Retrieving relevant Striver A2Z content…
          </p>
        </>
      ) : (
        <>
          <h2 className="loading-title">GENERATING ANSWER</h2>
          <p className="loading-subtitle">
            Grounding response in retrieved sources…
          </p>
        </>
      )}
    </section>
  );
}
