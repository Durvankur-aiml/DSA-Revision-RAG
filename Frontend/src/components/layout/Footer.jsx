import AlgoForgeLogo from "../brand/AlgoForgeLogo";

/**
 * Minimal footer — a quiet bookend for the shell.
 *
 * ALGOFORGE wordmark + tagline, and the one product promise that is
 * actually true: answers are grounded in the indexed Striver A2Z
 * course and sources deep-link to the exact timestamp. No fake links,
 * social icons, or legal filler.
 */
export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <p className="footer-brand">
          <span className="footer-mark" aria-hidden="true">
            <AlgoForgeLogo size={18} withGlow={false} />
          </span>
          ALGOFORGE
          <span className="footer-tagline">
            Navigate the world of algorithms.
          </span>
        </p>
        <p className="footer-note">
          Answers are grounded in the indexed Striver A2Z course. Sources link
          to the exact video timestamp.
        </p>
      </div>
    </footer>
  );
}
