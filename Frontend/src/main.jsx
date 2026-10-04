import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Self-hosted variable fonts (no network dependency, no layout shift
// from late font swaps). Registered family names:
//   'Inter Variable'          (wght 100-900)
//   'JetBrains Mono Variable' (wght 100-800)
// Tokens lead with these names and fall back to the previous stacks.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/index.css";

import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
