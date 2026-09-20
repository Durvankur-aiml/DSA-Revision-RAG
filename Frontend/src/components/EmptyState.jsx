/**
 * Landing / empty state shown before the first question.
 * Example chips submit directly (one click = one request).
 */
export default function EmptyState({ onExampleSelect }) {
  const examples = [
    'What is binary search?',
    'Explain binary search tree.',
    'What is the time complexity of merge sort?',
    'Explain two pointer technique.',
  ];

  return (
    <section className="empty-state" aria-labelledby="empty-heading">
      <p className="empty-eyebrow">ASK YOUR DSA MENTOR</p>
      <h1 id="empty-heading" className="empty-heading">
        Ask your DSA mentor.
      </h1>
      <p className="empty-subtitle">
        Search the Striver A2Z knowledge base and get grounded answers with
        video timestamps.
      </p>

      <div className="examples" role="list" aria-label="Example questions">
        <span className="examples-label">Try asking</span>
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            role="listitem"
            className="example-chip"
            onClick={() => onExampleSelect(example)}
          >
            {example}
          </button>
        ))}
      </div>
    </section>
  );
}
