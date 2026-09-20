import { useEffect, useRef } from 'react';
import Message from './Message';

/**
 * The session's message history. Newest message is scrolled into view.
 */
export default function Conversation({ messages }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages]);

  if (!messages.length) return null;

  return (
    <section className="conversation" aria-label="Conversation">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      <div ref={endRef} />
    </section>
  );
}
