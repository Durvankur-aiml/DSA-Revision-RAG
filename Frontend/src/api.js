/**
 * API client for the Mission Anthropic backend.
 *
 * Contract (backend is the source of truth):
 *   POST {VITE_API_URL}/ask    { question: string }
 *     -> 200 { answer: string, sources: Source[] }
 *     -> 400 | 404 | 422 | 502  { detail: string }
 *   GET  {VITE_API_URL}/health -> 200 { status: string }
 *
 * No streaming. One request, one complete JSON response.
 * No credentials or Gemini keys exist in the frontend.
 */

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/**
 * @typedef {Object} Source
 * @property {string} title
 * @property {number} timestamp  seconds into the video
 * @property {string} url        YouTube deep link including ?t=
 * @property {number} score      hybrid retrieval relevance (0..1)
 */

/**
 * @typedef {Object} AskResponse
 * @property {string} answer
 * @property {Source[]} sources
 */

/**
 * Maps an HTTP status from the backend to a user-facing message.
 * Backend `detail` strings are preferred when present.
 * Never exposes internals (stack traces, Gemini internals, paths).
 */
function errorMessageFor(status, detail) {
  switch (status) {
    case 400:
      return (
        detail ??
        'That question could not be processed. Please rephrase and try again.'
      );
    case 404:
      return (
        detail ??
        'No relevant information was found in the Striver A2Z knowledge base.'
      );
    case 422:
      return 'That request was not valid. Please adjust your question and try again.';
    case 502:
      return 'The AI service is temporarily unavailable. Please try again in a moment.';
    default:
      return detail ?? 'Something went wrong. Please try again.';
  }
}

/**
 * Ask the knowledge base a question.
 *
 * @param {string} question
 * @returns {Promise<AskResponse>}
 * @throws {Error} with a user-safe message when the request fails
 */
export async function askQuestion(question) {
  const trimmed = (question ?? '').trim();

  // Client-side guard; the backend remains authoritative.
  if (!trimmed) {
    throw new Error('Please enter a question before submitting.');
  }

  let response;
  try {
    response = await fetch(`${API_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: trimmed }),
    });
  } catch {
    // Network-level failure (backend down, CORS, DNS...).
    throw new Error('Could not reach the Mission Anthropic API. Is the backend running?');
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail =
      body && typeof body.detail === 'string' ? body.detail : null;
    throw new Error(errorMessageFor(response.status, detail));
  }

  if (!body || typeof body.answer !== 'string' || !Array.isArray(body.sources)) {
    throw new Error('The API returned an unexpected response format.');
  }

  return body;
}

/**
 * Check backend availability for the header status indicator.
 *
 * @returns {Promise<boolean>} true when /health responds 200
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
