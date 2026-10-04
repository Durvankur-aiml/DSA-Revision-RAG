import { useEffect, useRef, useState } from 'react';
import { ArrowUp } from 'lucide-react';

const MAX_HEIGHT = 160; // px; beyond this the textarea scrolls internally

/**
 * ALGOFORGE composer.
 *
 * - Auto-growing <textarea> with a hard max height (internal scroll
 *   beyond it) — long multi-line questions work as intended.
 * - Enter submits; Shift+Enter inserts a newline. During IME
 *   composition (Enter confirms the candidate), Enter does NOT submit.
 * - Empty/whitespace-only questions cannot be submitted; the backend
 *   remains authoritative (validate_question).
 * - Disabled while a request is in flight (no duplicate submissions).
 */
export default function QuestionInput({ onSubmit, disabled, autoFocus }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [autoFocus]);

  // Auto-grow: reset height, then clamp to MAX_HEIGHT so overflow
  // becomes internal scrolling.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  const canSubmit = !disabled && value.trim().length > 0;

  function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(value);
    setValue('');
  }

  function handleKeyDown(event) {
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
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
        <textarea
          id="question-input"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about algorithms, data structures, complexity…"
          autoComplete="off"
          disabled={disabled}
          aria-describedby="question-hint"
        />
        <button
          type="submit"
          className="submit-button"
          disabled={!canSubmit}
          aria-label={
            disabled
              ? 'Waiting for the current answer'
              : 'Send question'
          }
          title={canSubmit ? 'Send (Enter)' : undefined}
        >
          <ArrowUp size={17} strokeWidth={2.4} aria-hidden="true" />
        </button>
      </div>
      <p id="question-hint" className="input-hint">
        Press Enter to ask · Shift+Enter for a new line
      </p>
    </form>
  );
}
