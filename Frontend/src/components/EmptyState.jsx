import AlgoForgeLogo from './brand/AlgoForgeLogo';

/**
 * Landing / empty state shown before the first question.
 *
 * Hierarchy: brand wordmark -> tagline -> scope statement -> example
 * prompts. Examples submit directly (one click = one request).
 */
export default function EmptyState({ onExampleSelect }) {
  const examples = [
    'Explain binary search step by step',
    'When should I use sliding window?',
    'Explain the two pointer pattern',
    "What's the time complexity of merge sort?",
  ];

  return (
    <section className="empty-state" aria-labelledby="empty-heading">
      <div className="empty-logo" aria-hidden="true">
        <AlgoForgeLogo size={64} />
      </div>

      <h1 id="empty-heading" className="empty-heading">
        ALGOFORGE
      </h1>
      <p className="empty-tagline">Navigate the world of algorithms.</p>

      <p className="empty-scope">
        Ask questions about data structures, algorithms, patterns,
        complexity, or problems from the Striver A2Z knowledge base.
      </p>

      <div className="examples" role="list" aria-label="Example questions">
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
