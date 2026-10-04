import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import Message from './Message';

/**
 * Animate the window scroll to the bottom of the document.
 *
 * A manual rAF tween is used instead of window.scrollTo({ behavior:
 * 'smooth' }) because native smooth scrolling is unsupported in some
 * embedded webviews and disabled when the user prefers reduced motion.
 * The tween is cancelled the moment the user scrolls, so it never
 * hijacks manual scrolling, and it falls back to an instant jump when
 * the page is hidden or rAF is suspended (e.g. backgrounded webviews).
 */
function useScrollToBottom() {
  const rafRef = useRef(0);
  const watchdogRef = useRef(0);

  const cancel = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    if (watchdogRef.current) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = 0;
    }
  }, []);

  // User intent (wheel / touch) always wins over an in-flight tween.
  useEffect(() => {
    window.addEventListener('wheel', cancel, { passive: true });
    window.addEventListener('touchstart', cancel, { passive: true });
    return () => {
      window.removeEventListener('wheel', cancel);
      window.removeEventListener('touchstart', cancel);
      cancel();
    };
  }, [cancel]);

  return useCallback(() => {
    cancel();
    const target = document.documentElement.scrollHeight;
    const start = window.scrollY;
    const distance = target - start;
    if (distance <= 0) return;

    const instant = () => window.scrollTo(0, target);

    if (
      document.visibilityState === 'hidden' ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      instant();
      return;
    }

    const duration = Math.min(450, Math.max(180, distance * 0.15));
    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
    let startTime = null;
    let progressed = false;
    const step = (ts) => {
      if (startTime === null) startTime = ts;
      progressed = true;
      const progress = Math.min(1, (ts - startTime) / duration);
      window.scrollTo(0, start + distance * easeOutCubic(progress));
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = 0;
      }
    };
    rafRef.current = requestAnimationFrame(step);

    // Some embedded webviews suspend rAF while the tab is hidden. If the
    // tween has made no progress shortly after starting, arrive at the
    // target instantly instead of getting stuck partway.
    watchdogRef.current = setTimeout(() => {
      if (!progressed && rafRef.current) {
        cancel();
        instant();
      }
    }, 120);
  }, [cancel]);
}

/**
 * Distance from the bottom of the document, in pixels.
 */
function distanceFromBottom() {
  return (
    document.documentElement.scrollHeight -
    window.scrollY -
    window.innerHeight
  );
}

/**
 * The session's message history.
 *
 * Auto-scroll policy: the conversation is pinned to the bottom while
 * the user is already near it. If the user scrolls up to read older
 * content, the view is NOT hijacked back down — a "jump to latest"
 * affordance appears instead. Pinned scrolling resumes when the user
 * returns to the bottom.
 *
 * The pin decision is computed from live scroll geometry at the moment
 * new content arrives, not from cached scroll-event state, so it stays
 * correct even where scroll events are throttled. The document (window)
 * is the scroll container in this layout.
 */
export default function Conversation({ messages, isLoading, onRetry }) {
  const scrollToBottom = useScrollToBottom();
  const [pinned, setPinned] = useState(true);

  const updatePinned = useCallback(() => {
    setPinned(distanceFromBottom() < 120);
  }, []);

  // Keep the "jump to latest" affordance in sync while the user scrolls.
  useEffect(() => {
    window.addEventListener('scroll', updatePinned, { passive: true });
    updatePinned();
    return () => window.removeEventListener('scroll', updatePinned);
  }, [updatePinned]);

  // On new content: refresh the pin state from live geometry, then keep
  // the latest content visible only if the user was already near it.
  useEffect(() => {
    updatePinned();
    if (distanceFromBottom() < 120) scrollToBottom();
  }, [messages, isLoading, updatePinned, scrollToBottom]);

  if (!messages.length) return null;

  return (
    <>
      <section className="conversation" aria-label="Conversation">
        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            onRetry={message.failed ? onRetry : undefined}
          />
        ))}
      </section>

      {!pinned && (
        <button
          type="button"
          className="jump-latest"
          onClick={() => {
            scrollToBottom();
            updatePinned();
          }}
          aria-label="Jump to latest message"
          title="Jump to latest"
        >
          <ArrowDown size={15} aria-hidden="true" />
        </button>
      )}
    </>
  );
}
