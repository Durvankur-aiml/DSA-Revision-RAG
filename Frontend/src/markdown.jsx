/**
 * Minimal, dependency-free Markdown renderer for AI answers.
 *
 * WHY not a Markdown package: the backend answers use a small, known
 * subset of Markdown. Rendering it to React elements directly avoids
 * both a dependency and any HTML-injection surface: nothing here uses
 * dangerouslySetInnerHTML, and all text content is escaped by React
 * itself.
 *
 * Supported block syntax:
 *   # / ## / ### headings
 *   - / * bullets          1. ordered lists
 *   ``` fenced code        --- horizontal rules
 *   paragraphs
 * Inline: **bold**, *italic*, `code`, [text](https://url)
 *
 * Fenced code renders through the CodeBlock component (language label
 * + copy button). Links are restricted to https/http/mailto and open
 * safely in a new tab; anything else renders as plain text.
 */

import React from 'react';
import CodeBlock from './components/CodeBlock';

// ---------------------------------------------------------------------------
// Inline formatting -> React nodes
// ---------------------------------------------------------------------------

// Combined tokenizer: bold, italic, inline code, links.
const INLINE_PATTERN = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\[[^\]]+\]\((?:https?:\/\/|mailto:)[^\s)]+\))/g;
const LINK_PATTERN = /^\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)$/;

function renderInline(text, keyPrefix) {
  const pattern = new RegExp(INLINE_PATTERN);
  const nodes = [];
  let lastIndex = 0;
  let match;
  let i = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    const key = `${keyPrefix}-i${i++}`;

    const linkMatch = LINK_PATTERN.exec(token);
    if (linkMatch) {
      nodes.push(
        <a key={key} href={linkMatch[2]} target="_blank" rel="noopener noreferrer">
          {linkMatch[1]}
        </a>
      );
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('`')) {
      nodes.push(<code key={key} className="inline-code">{token.slice(1, -1)}</code>);
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

// ---------------------------------------------------------------------------
// Block parsing -> React elements
// ---------------------------------------------------------------------------

function parseBlocks(lines) {
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block: ``` optionally followed by a language tag.
    if (line.trimStart().startsWith('```')) {
      const language = line.trim().replace(/^```/, '').trim().toLowerCase();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1; // closing fence
      blocks.push({ type: 'code', lines: codeLines, language });
      continue;
    }

    // Horizontal rule
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      blocks.push({ type: 'hr' });
      i += 1;
      continue;
    }

    // Headings
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      blocks.push({ type: 'heading', level: heading[1].length, text: heading[2] });
      i += 1;
      continue;
    }

    // Unordered list (contiguous)
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
        i += 1;
      }
      blocks.push({ type: 'ul', items });
      continue;
    }

    // Ordered list (contiguous)
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ''));
        i += 1;
      }
      blocks.push({ type: 'ol', items });
      continue;
    }

    // Blank line
    if (line.trim() === '') {
      i += 1;
      continue;
    }

    // Paragraph (contiguous non-empty, non-special lines)
    const paragraphLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^\s*([-*]\s+|\d+[.)]\s+|#{1,4}\s+|```)/.test(lines[i]) &&
      !/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i])
    ) {
      paragraphLines.push(lines[i]);
      i += 1;
    }
    if (paragraphLines.length > 0) {
      blocks.push({ type: 'p', text: paragraphLines.join(' ') });
    }
  }

  return blocks;
}

export default function Markdown({ text }) {
  if (!text || typeof text !== 'string') {
    return null;
  }

  const blocks = parseBlocks(text.replace(/\r\n/g, '\n').split('\n'));

  return (
    <div className="markdown">
      {blocks.map((block, bi) => {
        const key = `b${bi}`;
        switch (block.type) {
          case 'heading': {
            const Tag =
              block.level === 1 ? 'h3' : block.level === 2 ? 'h4' : 'h5';
            return <Tag key={key}>{renderInline(block.text, key)}</Tag>;
          }
          case 'p':
            return <p key={key}>{renderInline(block.text, key)}</p>;
          case 'ul':
            return (
              <ul key={key}>
                {block.items.map((item, li) => (
                  <li key={`${key}-${li}`}>
                    {renderInline(item, `${key}-${li}`)}
                  </li>
                ))}
              </ul>
            );
          case 'ol':
            return (
              <ol key={key}>
                {block.items.map((item, li) => (
                  <li key={`${key}-${li}`}>
                    {renderInline(item, `${key}-${li}`)}
                  </li>
                ))}
              </ol>
            );
          case 'code':
            return (
              <CodeBlock
                key={key}
                code={block.lines.join('\n')}
                language={block.language}
              />
            );
          case 'hr':
            return <hr key={key} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
