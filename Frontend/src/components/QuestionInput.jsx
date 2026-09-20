import { useEffect, useRef, useState } from 'react';

/**
 * Premium question input.
 * - Enter submits; Shift+Enter inserts a newline
 * - disabled while a request is in flight (prevents duplicate submits)
 * - whitespace-only input cannot be submitted (backend stays authoritative)
 */
export default function QuestionInput({ onSubmit, disabled, autoFocus }) {
  const [value, setValue] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const canSubmit = !disabled && value.trim().length > 0;

  function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(value);
    setValue('');
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  }

  return (
    <form className="question-form" onSubmit={handleSubmit} role="search">
      <label className="visually-hidden" htmlFor="question-input">
        Your DSA question
      </label>
      <div className={`input-shell ${disabled ? 'input-disabled' : ''}`}>
        <input
          id="question-input"
          ref={inputRef}
          type="text"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="What do you want to learn?"
          autoComplete="off"
          disabled={disabled}
          aria-describedby="question-hint"
        />
        <button
          type="submit"
          className="submit-button"
          disabled={!canSubmit}
          aria-label="Ask the knowledge engine"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M5 12h14M13 6l6 6-6 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      <p id="question-hint" className="input-hint">
        Press Enter to ask · Shift+Enter for a new line
      </p>
    </form>
  );
}
