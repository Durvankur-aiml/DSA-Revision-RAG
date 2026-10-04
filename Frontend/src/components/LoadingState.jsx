import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Loading state shown while POST /ask is in flight.
 *
 * This is a UI indicator only: it NEVER fabricates partial answer
 * text or fakes token streaming. The two phases are honest pacing
 * while the single request completes (retrieval is fast post-
 * optimization; generation dominates, so the phase switch at 1.5s
 * reflects where the time actually goes).
 */
export default function LoadingState() {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setPhase(1), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section className="loading-state" aria-live="polite" aria-busy="true">
      <div className="loading-spinner" aria-hidden="true" />

      <AnimatePresence mode="wait" initial={false}>
        {phase === 0 ? (
          <motion.div
            key="searching"
            className="loading-copy"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            <h2 className="loading-title">Searching the knowledge base…</h2>
            <p className="loading-subtitle">
              Retrieving relevant Striver A2Z content
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="generating"
            className="loading-copy"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            <h2 className="loading-title">Generating your answer…</h2>
            <p className="loading-subtitle">
              Grounding the response in retrieved sources
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
