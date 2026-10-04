import { useId } from "react";

/**
 * ALGOFORGE logo mark — the approved "geometric A + algorithm nodes".
 *
 * Geometry and colors follow the approved branding sheet (panel 5/6):
 * a letter-A formed by two outer legs and an apex node, a smaller
 * internal crossbar node, three outer nodes, connected by rounded
 * strokes, with a violet -> cyan accent treatment and a soft glow.
 *
 *   apex     (128, 64)     r 26
 *   left     (64, 192)     r 26
 *   right    (192, 192)    r 26
 *   crossbar (192, 128)    r 16   (internal node)
 *   legs: apex->left, apex->right, left->crossbar
 *
 * Fills use the approved gradient (#A855F7 -> #22D3EE); strokes read
 * as violet -> blue/cyan across the A. Glow is a soft feGaussianBlur
 * layered beneath the crisp mark.
 *
 * Props:
 *   size      rendered square size in px (default 40)
 *   withGlow  include the soft glow layer (default true)
 *   title     accessible label; omit for purely decorative use
 *
 * Gradient/filter IDs are namespaced per component instance so any
 * number of logos can coexist on one page without SVG ID collisions.
 */
export default function AlgoForgeLogo({ size = 40, withGlow = true, title }) {
  const uid = useId();
  const gradId = `af-grad-${uid}`;
  const glowId = `af-glow-${uid}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 256 256"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#A855F7" />
          <stop offset="100%" stopColor="#22D3EE" />
        </linearGradient>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="10" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g filter={withGlow ? `url(#${glowId})` : undefined}>
        {/* Strokes: the A silhouette + the internal path */}
        <path
          d="M64 192 L128 64 L192 192"
          stroke={`url(#${gradId})`}
          strokeWidth="18"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M64 192 L192 128"
          stroke={`url(#${gradId})`}
          strokeWidth="10"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Nodes: three outer + one internal */}
        <circle cx="128" cy="64" r="26" fill={`url(#${gradId})`} />
        <circle cx="64" cy="192" r="26" fill={`url(#${gradId})`} />
        <circle cx="192" cy="192" r="26" fill={`url(#${gradId})`} />
        <circle cx="192" cy="128" r="16" fill={`url(#${gradId})`} />
      </g>
    </svg>
  );
}
