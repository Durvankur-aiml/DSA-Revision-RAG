import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

/**
 * Fenced code block with an optional language label and copy button.
 *
 * - Horizontal scrolling for long lines (no wrapping).
 * - Copy uses the Clipboard API with a transient "Copied" state and
 *   a fallback for non-secure contexts (http://localhost qualifies as
 *   secure in Chromium, but file:// or plain http:// LAN may not).
 * - The button has an accessible name and announces success.
 */
export default function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false);

  async function copyText(text) {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch {
        // Fall through to the legacy path (focus/permission hiccups).
      }
    }
    // Fallback for non-secure contexts and transient Clipboard API
    // failures.
    const el = document.createElement('textarea');
    el.value = text;
    el.setAttribute('readonly', '');
    el.style.position = 'fixed';
    el.style.opacity = '0';
    document.body.appendChild(el);
    el.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch {
      ok = false;
    }
    document.body.removeChild(el);
    return ok;
  }

  async function handleCopy() {
    const ok = await copyText(code ?? '');
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    }
  }

  return (
    <figure className="code-block">
      <figcaption className="code-block-bar">
        <span className="code-block-language">
          {language || 'code'}
        </span>
        <button
          type="button"
          className="code-copy-button"
          onClick={handleCopy}
          aria-label={copied ? 'Code copied' : 'Copy code to clipboard'}
          aria-live="polite"
          title={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? (
            <Check size={13} aria-hidden="true" />
          ) : (
            <Copy size={13} aria-hidden="true" />
          )}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </figcaption>
      <pre>
        <code>{code}</code>
      </pre>
    </figure>
  );
}
