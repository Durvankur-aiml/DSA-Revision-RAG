"""ALGOFORGE Deterministic Agent Router.

Classifies incoming user questions into canonical intents and determines
the appropriate retrieval depth, reranking policy, and response mode
BEFORE execution, adding zero LLM latency and sub-millisecond overhead.
"""

import re
import time
from typing import Optional, Tuple

import obs
from .models import QueryIntent, AgentDecision


# ---------------------------------------------------------------------------
# Compiled Heuristics & Vocabulary
# ---------------------------------------------------------------------------

# Core DSA domain vocabulary (data structures, algorithms, problem names)
_DSA_TERMS = re.compile(
    r"\b("
    # Data structures
    r"arrays?|matrix|matrices|strings?|linked\s*lists?|doubly\s*linked\s*list|"
    r"stacks?|queues?|deque|priority\s*queue|heaps?|min\s*heap|max\s*heap|"
    r"trees?|binary\s*trees?|bst|binary\s*search\s*tree|tries?|avl|red\s*black|"
    r"graphs?|dag|directed\s*graph|undirected\s*graph|disjoint\s*sets?|dsu|"
    r"hash\s*maps?|hash\s*tables?|hashing|"
    # Algorithms & Techniques
    r"binary\s*search|linear\s*search|lower\s*bound|upper\s*bound|"
    r"two\s*pointers?|sliding\s*windows?|recursion|backtracking|"
    r"dynamic\s*programming|\bdp\b|memoization|tabulation|"
    r"greedy|kadane|sorting|bubble\s*sort|selection\s*sort|insertion\s*sort|"
    r"merge\s*sort|quick\s*sort|cyclic\s*sort|topological\s*sort|kahn|"
    r"bfs|dfs|breadth\s*first|depth\s*first|dijkstra|bellman\s*ford|"
    r"floyd\s*warshall|prim|kruskal|kosaraju|tarjan|bit\s*manipulation|"
    r"time\s*complexity|space\s*complexity|big\s*o|"
    # Popular Striver A2Z DSA problems
    r"two\s*sum|three\s*sum|3\s*sum|4\s*sum|trapping\s*rain\s*water|"
    r"next\s*permutation|pascal\s*triangle|set\s*matrix\s*zeroes|"
    r"subarrays?|subsequences?|longest\s*consecutive|"
    r"reverse\s*linked\s*list|detect\s*cycle|middle\s*of\s*linked\s*list|"
    r"lru\s*cache|lfu\s*cache|sliding\s*window\s*maximum|min\s*stack|"
    r"balanced\s*parenthes[ei]s|diameter\s*of\s*binary\s*tree|lowest\s*common\s*ancestor|\blca\b|"
    r"morris\s*traversal|vertical\s*order|top\s*view|bottom\s*view|boundary\s*traversal|"
    r"course\s*schedule|alien\s*dictionary|word\s*ladder|rotting\s*oranges|"
    r"number\s*of\s*islands|flood\s*fill|clone\s*graph|"
    r"0/1\s*knapsack|knapsack|coin\s*change|longest\s*common\s*subsequence|\blcs\b|"
    r"longest\s*increasing\s*subsequence|\blis\b|edit\s*distance|matrix\s*chain\s*multiplication|\bmcm\b|"
    r"subset\s*sum|partition\s*equal\s*subset|n\s*queens|sudoku\s*solver|"
    r"allocate\s*books|painters?\s*partition|koko\s*eating\s*bananas|"
    r"buy\s*and\s*sell\s*stocks?"
    r")\b",
    re.IGNORECASE,
)

# Out-of-scope non-DSA domains
_OUT_OF_SCOPE_TERMS = re.compile(
    r"\b("
    r"weather|forecast|rain|rainy|sunny|temperature|climate|"
    r"recipe|cooking|cake|pizza|burger|pasta|bake|cuisine|food|"
    r"cricket|football|soccer|basketball|ipl|fifa|messi|ronaldo|tennis|virat\s*kohli|"
    r"president|prime\s*minister|parliament|election|capital\s*of|monument|"
    r"movie|hollywood|bollywood|song|lyrics|singer|actor|actress|celebrity|"
    r"quantum\s*chromodynamics|photosynthesis|mitochondria|black\s*hole|astronomy|chemical\s*reaction|"
    r"bitcoin|ethereum|crypto|stock\s*market|dow\s*jones|nasdaq"
    r")\b",
    re.IGNORECASE,
)

_CODE_DEBUG_TERMS = re.compile(
    r"\b("
    r"why\s+does\s+(this|my|the)?\s*[\w\s]{0,35}?(code|solution|implementation|snippet)\s+(fail|give\s+error|crash|segfault|produce\s+wrong|not\s+work)|"
    r"(code|solution|implementation|program)\s+(fails?|crashes?|segfaults?|gives?\s+error|gives?\s+tle)|"
    r"debug\b|fix\s+(this|my)?\s*code|what\s+is\s+wrong\s+with\s+(this|my)?\s*code|"
    r"bugs?\b|what\s+is\s+(the\s+)?bug|find\s+(the\s+)?bug|"
    r"syntax\s*error|runtime\s*error|time\s*limit\s*exceeded|\btle\b|\bmle\b|"
    r"segmentation\s*fault|null\s*pointer|nullpointerexception|infinite\s*loop\s+in"
    r")\b",
    re.IGNORECASE,
)

_REVISION_TERMS = re.compile(
    r"\b("
    r"revision|revise|quick\s+summary|summary\s+of|cheat\s*sheet|cheat-sheet|"
    r"recap|refresher|brush\s*up|quick\s+overview\s+of|one\s*shot"
    r")\b",
    re.IGNORECASE,
)

_COMPARISON_TERMS = re.compile(
    r"\b("
    r"vs|versus|difference\s+between|compare|comparison\s+between|"
    r"pros\s+and\s+cons\s+of|which\s+is\s+better|tradeoffs?\s+between|"
    r"when\s+to\s+use\s+.*\s+(over|or)\s+"
    r")\b",
    re.IGNORECASE,
)

_EXPLANATION_TERMS = re.compile(
    r"\b("
    r"explain\s+.*\s+step\s+by\s+step|step\s+by\s+step|detailed\s+explanation|"
    r"walk\s*through|dry\s*run|how\s+does\s+.*\s+work|trace\s+through|"
    r"in-depth\s+explanation"
    r")\b",
    re.IGNORECASE,
)

_PROBLEM_SOLVING_TERMS = re.compile(
    r"\b("
    r"how\s+(do|can)\s+(i|we)\s+solve|solve\b|solution\s+for|approach\s+for|"
    r"how\s+to\s+solve|algorithm\s+(to|for|of)\s+|optimal\s+approach|"
    r"brute\s+force\s+(approach|for)|intuition\s+(behind|for)|"
    r"write\s+(a\s+)?code\s+for|program\s+to\s+solve"
    r")\b",
    re.IGNORECASE,
)

_CONCEPT_TERMS = re.compile(
    r"\b("
    r"what\s+is|define\b|concept\s+of|introduction\s+to|meaning\s+of|"
    r"overview\s+of|what\s+are|what\s+do\s+you\s+mean\s+by|"
    r"explain\s+the\s+concept|properties\s+of"
    r")\b",
    re.IGNORECASE,
)

_FOLLOW_UP_TERMS = re.compile(
    r"\b("
    r"why\s+does\s+that\s+work|why\s+did\s+that\s+happen|"
    r"explain\s+the\s+(first|second|third|last|previous)\s+step|"
    r"what\s+did\s+you\s+mean\s+by\s+that|can\s+you\s+clarify\s+that|"
    r"tell\s+me\s+more\s+about\s+that|why\s+not\s+the\s+other\s+way|"
    r"what\s+about\s+earlier|can\s+you\s+elaborate\s+on\s+that|"
    r"what\s+if\s+we\s+do\s+that\s+instead|why\s+so\b|how\s+come\b"
    r")\b",
    re.IGNORECASE,
)


class AgentRouter:
    """Production Agent Router for deterministic intent classification."""

    @staticmethod
    def route(query: str, request_id: Optional[str] = None) -> AgentDecision:
        """Analyze query and return an AgentDecision.

        Parameters
        ----------
        query : str
            Raw user query string.
        request_id : Optional[str]
            Tracing request ID.

        Returns
        -------
        AgentDecision
            Deterministic routing configuration.
        """
        t0 = time.monotonic()
        text = str(query or "").strip()
        t_lower = text.lower()

        # Handle empty/whitespace input
        if not text:
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.OUT_OF_SCOPE,
                retrieval_required=False,
                retrieval_depth=0,
                reranking_required=False,
                response_mode="fast_fail",
                confidence=1.0,
                reason="Empty or whitespace query provided.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        has_dsa = bool(_DSA_TERMS.search(t_lower))
        has_out_of_scope = bool(_OUT_OF_SCOPE_TERMS.search(t_lower))

        # Check for stock buying special case in DSA (Buy and Sell Stock)
        if "stock" in t_lower and ("buy" in t_lower or "sell" in t_lower or "dp" in t_lower):
            has_out_of_scope = False
            has_dsa = True

        # 1. OUT_OF_SCOPE: Clear non-DSA topic without DSA keywords
        if has_out_of_scope and not has_dsa:
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.OUT_OF_SCOPE,
                retrieval_required=False,
                retrieval_depth=0,
                reranking_required=False,
                response_mode="fast_fail",
                confidence=0.98,
                reason="Query contains clear non-DSA topics outside of Striver A2Z curriculum.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 2. FOLLOW_UP: Conversational continuation referring to prior turns
        if _FOLLOW_UP_TERMS.search(t_lower):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.FOLLOW_UP,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="conversational",
                confidence=0.95,
                reason="Query references previous step or conversational antecedent.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 3. CODE_DEBUG: Error diagnosis or code debugging
        has_code_syntax = bool(re.search(r"```|def\s+|int\s+main|class\s+Solution|while\s*\(|nullptr", text))
        if _CODE_DEBUG_TERMS.search(t_lower) or (has_code_syntax and ("fail" in t_lower or "error" in t_lower or "bug" in t_lower)):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.CODE_DEBUG,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="code_debug",
                confidence=0.95,
                reason="Query asks to diagnose or debug code implementation.",
                suggested_prompt_prefix="Analyze the code bug and explain the logical or syntax error.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 4. REVISION: Summary or recap
        if _REVISION_TERMS.search(t_lower):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.REVISION,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="concise_revision",
                confidence=0.95,
                reason="User requested concise revision/recap of topic.",
                suggested_prompt_prefix="Provide a structured, high-yield summary for quick revision.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 5. COMPARISON: Contrasting approaches or data structures
        if _COMPARISON_TERMS.search(t_lower):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.COMPARISON,
                retrieval_required=True,
                retrieval_depth=8,  # Slightly broader context for comparison
                reranking_required=True,
                response_mode="comparison",
                confidence=0.95,
                reason="Query compares two or more algorithmic techniques or data structures.",
                suggested_prompt_prefix="Compare the algorithms highlighting time/space trade-offs and use cases.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 6. EXPLANATION: Step-by-step walkthrough or dry run
        if _EXPLANATION_TERMS.search(t_lower):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.EXPLANATION,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="step_by_step",
                confidence=0.92,
                reason="User requested a step-by-step explanation or trace.",
                suggested_prompt_prefix="Explain step by step with a clear dry-run walkthrough.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 7. PROBLEM_SOLVING: Specific problem approach or algorithm
        if _PROBLEM_SOLVING_TERMS.search(t_lower):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.PROBLEM_SOLVING,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="problem_solving",
                confidence=0.90,
                reason="Query asks for approach, intuition, or code solution to a DSA problem.",
                suggested_prompt_prefix="Explain the intuition, brute force, and optimal solution clearly.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 8. CONCEPT: Definition or concept overview
        if _CONCEPT_TERMS.search(t_lower) or (has_dsa and len(text.split()) <= 5):
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.CONCEPT,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="conceptual",
                confidence=0.88,
                reason="Query asks for foundational concept definition or overview.",
                suggested_prompt_prefix="Define the concept clearly with core principles and examples.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 9. Fallback: If DSA keywords are found, route to standard retrieval
        if has_dsa:
            duration = time.monotonic() - t0
            decision = AgentDecision(
                intent=QueryIntent.PROBLEM_SOLVING,
                retrieval_required=True,
                retrieval_depth=6,
                reranking_required=True,
                response_mode="standard",
                confidence=0.80,
                reason="DSA keywords recognized in query; routed to standard hybrid retrieval.",
            )
            AgentRouter._log_decision(decision, duration, request_id)
            return decision

        # 10. Fallback: Completely unclassifiable non-DSA query
        duration = time.monotonic() - t0
        decision = AgentDecision(
            intent=QueryIntent.OUT_OF_SCOPE,
            retrieval_required=False,
            retrieval_depth=0,
            reranking_required=False,
            response_mode="fast_fail",
            confidence=0.75,
            reason="No DSA keywords or programming constructs detected.",
        )
        AgentRouter._log_decision(decision, duration, request_id)
        return decision

    @staticmethod
    def _log_decision(decision: AgentDecision, duration: float, request_id: Optional[str]) -> None:
        """Record observability metrics for routing without raising."""
        try:
            extra = (
                f"intent={decision.intent.value} "
                f"retrieval={decision.retrieval_required} "
                f"rerank={decision.reranking_required} "
                f"mode={decision.response_mode}"
            )
            if request_id:
                extra += f" req_id={request_id}"
            obs.record_stage("agent_routing", duration, extra=extra)

            if hasattr(obs, "set_agent_meta"):
                obs.set_agent_meta(
                    intent=decision.intent.value,
                    response_mode=decision.response_mode,
                    retrieval_required=decision.retrieval_required,
                    reranking_required=decision.reranking_required,
                )
        except Exception:
            pass


def route_query(query: str, request_id: Optional[str] = None) -> AgentDecision:
    """Convenience helper to route a query through the AgentRouter."""
    return AgentRouter.route(query, request_id=request_id)
