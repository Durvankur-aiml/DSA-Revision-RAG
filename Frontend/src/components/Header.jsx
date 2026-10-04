import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { checkHealth } from "../api";
import AlgoForgeLogo from "./brand/AlgoForgeLogo";

/**
 * ALGOFORGE application header.
 *
 * Brand: ALGOFORGE — "Navigate the world of algorithms."
 * The status pill reflects ONLY /health reachability (backend up/down).
 * It makes no claim about model availability, retrieval, or answer
 * quality — the health endpoint verifies none of those.
 *
 * A single subtle border/shadow appears once the page is scrolled
 * so the sticky header separates from content without heavy chrome.
 */
export default function Header() {
  const [online, setOnline] = useState(null); // null = checking
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    let cancelled = false;
    checkHealth().then((ok) => {
      if (!cancelled) setOnline(ok);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.header
      className={`header ${scrolled ? "header-scrolled" : ""}`}
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <AlgoForgeLogo size={30} />
        </span>
        <span className="brand-text">
          <span className="brand-name">ALGOFORGE</span>
          <span className="brand-subtitle">
            Navigate the world of algorithms.
          </span>
        </span>
      </div>

      <div
        className={`status ${online === true ? "status-online" : ""} ${
          online === false ? "status-offline" : ""
        }`}
        role="status"
        aria-live="polite"
        title={
          online === true
            ? "Backend reachable (GET /health)"
            : online === false
              ? "Backend unreachable (GET /health failed)"
              : "Checking backend…"
        }
      >
        <span className="status-dot" aria-hidden="true" />
        {online === null && "CHECKING API…"}
        {online === true && "API ONLINE"}
        {online === false && "API OFFLINE"}
      </div>
    </motion.header>
  );
}
