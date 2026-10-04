"""Build Data/eval/golden.jsonl from the human-curated question catalog.

Every question below was authored by hand from the actual 315-video
Striver A2Z course structure (Data/Metadata is the single source of
truth for video ids). This script is the traceability layer:

1. TITLE_KEYS maps eval_id -> unique substring of the expected video's
   metadata title. The script resolves the substring to the FULL
   metadata set and fails loudly if zero or multiple videos match,
   so no golden record can reference a guessed video id.
2. For "exact" chunk_certainty cases the script scans the video's
   transcript and lists the chunk indexes whose text matches the
   pattern(s) given in CHUNK_HINTS. All matching chunks are verified
   to exist in the embedding file. If no chunk matches, the case is
   downgraded to video-level certainty (never fabricated).
3. Validation (eval.dataset.validate_case) runs before writing.

Re-run any time: output is deterministic for the same Data/ tree.
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REPO_ROOT = BASE_DIR
METADATA_DIR = os.path.join(REPO_ROOT, "Data", "Metadata")
TRANSCRIPT_DIR = os.path.join(REPO_ROOT, "Data", "Transcripts")
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "Data", "embeddings")
OUT_PATH = os.path.join(REPO_ROOT, "Data", "eval", "golden.jsonl")


# ---------------------------------------------------------------------------
# Question catalog — hand-authored, traceable by title substring.
# ---------------------------------------------------------------------------
# (eval_id, question, title_substring, section, question_type, certainty)

CATALOG = [
    # ===================== 1. Binary Search =====================
    ("eval_001", "Explain binary search step by step.",
     "Binary Search Introduction", "Binary Search", "exact_topic", "video"),
    ("eval_002", "What is lower bound and upper bound in binary search?",
     "Lower Bound and Upper Bound", "Binary Search", "concept", "video"),
    ("eval_003", "How do I search in a rotated sorted array?",
     "Search Element in Rotated Sorted Array - I", "Binary Search",
     "algorithm", "video"),
    ("eval_004", "Find the minimum element in a rotated sorted array.",
     "Minimum in Rotated Sorted Array", "Binary Search", "algorithm", "video"),
    ("eval_005", "How does binary search work on the answer space, like Koko eating bananas?",
     "Koko Eating Bananas", "Binary Search", "algorithm", "video"),
    ("eval_006", "How do I allocate books to minimize the maximum pages a student gets?",
     "Allocate Books", "Binary Search", "algorithm", "video"),
    ("eval_007", "What is the time complexity of binary search and why?",
     "Binary Search Introduction", "Binary Search", "complexity", "video"),
    ("eval_008", "How does the painter's partition problem relate to split array largest sum?",
     "Painter's Partition and Split Array", "Binary Search", "concept", "video"),
    ("eval_009", "How do I find the square root of a number using binary search?",
     "Sqrt of a number using Binary Search", "Binary Search", "algorithm", "video"),
    ("eval_010", "How can I search in a 2D matrix efficiently?",
     "Search in a 2D Matrix - I |", "Binary Search", "algorithm", "video"),

    # ===================== 2. Sliding Window / Two Pointers =====================
    ("eval_011", "Explain the sliding window technique and when to use it.",
     "Introduction to Sliding Window and 2 Pointers",
     "Sliding Window", "exact_topic", "video"),
    ("eval_012", "How does the longest substring without repeating characters work?",
     "Longest Substring Without Repeating Characters", "Sliding Window",
     "algorithm", "video"),
    ("eval_013", "Solve max consecutive ones with at most k flips.",
     "Max Consecutive Ones III", "Sliding Window", "algorithm", "video"),
    ("eval_014", "What is the sliding window maximum problem and its optimal approach?",
     "Sliding Window Maximum", "Sliding Window", "algorithm", "video"),
    ("eval_015", "How do I find the longest subarray with sum K?",
     "Longest Subarray with sum K", "Sliding Window", "algorithm", "video"),
    ("eval_016", "Explain binary subarrays with sum equal to goal.",
     "Binary Subarrays With Sum", "Sliding Window", "algorithm", "video"),
    ("eval_017", "How does fruit into baskets work as a sliding window problem?",
     "Fruit Into Baskets", "Sliding Window", "algorithm", "video"),
    ("eval_018", "What is the minimum window substring approach?",
     "Minimum Window Substring", "Sliding Window", "algorithm", "video"),
    ("eval_019", "How do I count nice subarrays with k odd numbers?",
     "Nice subarrays", "Sliding Window", "algorithm", "video"),
    ("eval_020", "Longest repeating character replacement — explain the approach.",
     "Longest Repeating Character Replacement", "Sliding Window",
     "algorithm", "video"),

    # ===================== 3. Arrays (classic problems) =====================
    ("eval_021", "Explain Kadane's algorithm for maximum subarray sum.",
     "Kadane's Algorithm", "Arrays", "exact_topic", "video"),
    ("eval_022", "What is the Dutch national flag approach for sort an array of 0s 1s and 2s?",
     "Sort an array of 0's 1's & 2's", "Arrays", "algorithm", "video"),
    ("eval_023", "Solve the 3Sum problem with the optimal approach.",
     "3 Sum | Brute", "Arrays", "algorithm", "video"),
    ("eval_024", "Solve the 4Sum problem.",
     "4 Sum | Brute", "Arrays", "algorithm", "video"),
    ("eval_025", "How does Moore's voting algorithm find the majority element?",
     "Majority Element I | Brute", "Arrays", "algorithm", "video"),
    ("eval_026", "How do I set matrix zeroes in O(1) space?",
     "Set Matrix Zeroes", "Arrays", "algorithm", "video"),
    ("eval_027", "Explain the merge overlapping intervals problem.",
     "Merge Overlapping Intervals", "Arrays", "algorithm", "video"),
    ("eval_028", "What is the optimal way to rotate a matrix by 90 degrees?",
     "Rotate Matrix/Image by 90 Degrees", "Arrays", "algorithm", "video"),
    ("eval_029", "How does the next permutation algorithm work?",
     "Next Permutation", "Arrays", "algorithm", "video"),
    ("eval_030", "Find the longest consecutive sequence in an array.",
     "Longest Consecutive Sequence", "Arrays", "algorithm", "video"),

    # ===================== 4. Linked List =====================
    ("eval_031", "How do I reverse a linked list?",
     "Reverse a LinkedList", "Linked List", "algorithm", "video"),
    ("eval_032", "Find the middle element of a linked list.",
     "middle element of the LinkedList", "Linked List", "algorithm", "video"),
    ("eval_033", "How does Floyd's cycle detection work for linked lists?",
     "Detect a loop or cycle in LinkedList", "Linked List", "concept", "video"),
    ("eval_034", "How do I find the starting point of a loop in a linked list?",
     "starting point of the Loop", "Linked List", "algorithm", "video"),
    ("eval_035", "Explain the LRU cache implementation.",
     "Implement LRU Cache", "Linked List", "implementation", "video"),
    ("eval_036", "How does merge sort work on a linked list?",
     "Sort a Linked List | Merge Sort", "Linked List", "algorithm", "video"),
    ("eval_037", "What is the approach for flattening a linked list?",
     "Flattening a LinkedList", "Linked List", "algorithm", "video"),
    ("eval_038", "Check if a linked list is a palindrome.",
     "LinkedList is Palindrome", "Linked List", "algorithm", "video"),

    # ===================== 5. Stack and Queue =====================
    ("eval_039", "Explain the next greater element problem.",
     "L5. Next Greater Element", "Stack and Queue", "algorithm", "video"),
    ("eval_040", "How does the largest rectangle in histogram problem work?",
     "Largest Rectangle in Histogram", "Stack and Queue", "algorithm", "video"),
    ("eval_041", "What is the stock span problem?",
     "Stock Span Problem", "Stack and Queue", "algorithm", "video"),
    ("eval_042", "How do I implement a min stack?",
     "Implement Min Stack", "Stack and Queue", "implementation", "video"),
    ("eval_043", "Explain the trapping rainwater problem.",
     "Trapping Rainwater", "Stack and Queue", "algorithm", "video"),
    ("eval_044", "How does the celebrity problem work with a stack?",
     "The Celebrity Problem", "Stack and Queue", "algorithm", "video"),

    # ===================== 6. Binary Trees =====================
    ("eval_045", "Explain the tree traversals: inorder, preorder and postorder.",
     "Binary Tree Traversals in Binary Tree", "Binary Trees", "concept", "video"),
    ("eval_046", "How do I find the diameter of a binary tree?",
     "Diameter of Binary Tree", "Binary Trees", "algorithm", "video"),
    ("eval_047", "How is the lowest common ancestor of a binary tree found?",
     "Lowest Common Ancestor in Binary Tree", "Binary Trees", "algorithm", "video"),
    ("eval_048", "What is the maximum path sum in a binary tree?",
     "Maximum Path Sum in Binary Tree", "Binary Trees", "algorithm", "video"),
    ("eval_049", "How does the top view of a binary tree work?",
     "Top View of Binary Tree", "Binary Trees", "algorithm", "video"),
    ("eval_050", "Explain Morris traversal for inorder without recursion or stack.",
     "Morris Traversal", "Binary Trees", "algorithm", "video"),
    ("eval_051", "How do I check if a binary tree is balanced?",
     "Check for Balanced Binary Tree", "Binary Trees", "algorithm", "video"),
    ("eval_052", "What is the zig-zag or spiral level order traversal?",
     "Zig-Zag or Spiral Traversal", "Binary Trees", "algorithm", "video"),

    # ===================== 7. BST =====================
    ("eval_053", "What is a binary search tree and what are its properties?",
     "Introduction to Binary Search Tree", "BST", "exact_topic", "video"),
    ("eval_054", "How do I delete a node from a BST?",
     "Delete a Node in Binary Search Tree", "BST", "algorithm", "video"),
    ("eval_055", "Explain the inorder successor and predecessor in a BST.",
     "Inorder Successor/Predecessor in BST", "BST", "algorithm", "video"),
    ("eval_056", "How does two sum work inside a BST?",
     "Two Sum In BST", "BST", "algorithm", "video"),
    ("eval_057", "How do I validate whether a binary tree is a BST?",
     "tree is a BST or BT", "BST", "concept", "video"),
    ("eval_058", "Find the Kth smallest element in a BST.",
     "K-th Smallest/Largest Element in BST", "BST", "algorithm", "video"),

    # ===================== 8. Graphs =====================
    ("eval_059", "Explain BFS traversal in graphs.",
     "Breadth-First Search (BFS)", "Graphs", "exact_topic", "video"),
    ("eval_060", "Explain DFS traversal in graphs.",
     "Depth-First Search (DFS)", "Graphs", "exact_topic", "video"),
    ("eval_061", "How does Dijkstra's algorithm work?",
     "Dijkstra's Algorithm - Using Priority Queue", "Graphs", "algorithm", "video"),
    ("eval_062", "What is the number of islands problem and its approach?",
     "Number of Islands | Number of Connected Components", "Graphs",
     "algorithm", "video"),
    ("eval_063", "How does topological sort work with Kahn's algorithm?",
     "G-22. Kahn's Algorithm", "Graphs", "algorithm", "video"),
    ("eval_064", "Explain the disjoint set union data structure.",
     "Disjoint Set | Union by Rank", "Graphs", "concept", "video"),
    ("eval_065", "How does Kruskal's algorithm find the minimum spanning tree?",
     "Kruskal's Algorithm", "Graphs", "algorithm", "video"),
    ("eval_066", "What is the Bellman Ford algorithm used for?",
     "Bellman Ford Algorithm", "Graphs", "concept", "video"),
    ("eval_067", "How do I detect a cycle in a directed graph?",
     "Detect cycle in a directed graph", "Graphs", "algorithm", "video"),
    ("eval_068", "Explain the rotten oranges problem.",
     "Rotten Oranges", "Graphs", "algorithm", "video"),
    ("eval_069", "How does the word ladder problem work as a graph problem?",
     "Word Ladder - I", "Graphs", "algorithm", "video"),
    ("eval_070", "What is Kosaraju's algorithm for strongly connected components?",
     "Strongly Connected Components", "Graphs", "algorithm", "video"),

    # ===================== 9. Dynamic Programming =====================
    ("eval_071", "What is dynamic programming? Explain memoization and tabulation.",
     "Introduction to Dynamic Programming", "Dynamic Programming",
     "exact_topic", "video"),
    ("eval_072", "Explain the climbing stairs problem as 1D DP.",
     "Climbing Stairs | Learn How to Write 1D", "Dynamic Programming",
     "algorithm", "video"),
    ("eval_073", "How does the 0/1 knapsack work?",
     "0/1 Knapsack", "Dynamic Programming", "algorithm", "video"),
    ("eval_074", "Explain the house robber problem.",
     "Maximum Sum of Non-Adjacent Elements", "Dynamic Programming",
     "algorithm", "video"),
    ("eval_075", "What is the longest common subsequence approach?",
     "DP 25. Longest Common Subsequence", "Dynamic Programming",
     "algorithm", "video"),
    ("eval_076", "How does edit distance work?",
     "Edit Distance", "Dynamic Programming", "algorithm", "video"),
    ("eval_077", "Explain the coin change minimum coins problem.",
     "Minimum Coins", "Dynamic Programming", "algorithm", "video"),
    ("eval_078", "What is the longest increasing subsequence approach?",
     "Longest Increasing Subsequence | Binary Search", "Dynamic Programming",
     "algorithm", "video"),
    ("eval_079", "How does the grid unique paths DP work?",
     "Grid Unique Paths", "Dynamic Programming", "algorithm", "video"),
    ("eval_080", "Explain the partition equal subset sum problem.",
     "Partition Equal Subset Sum", "Dynamic Programming", "algorithm", "video"),
    ("eval_081", "What is the difference between DP 32 distinct subsequences and distinct palindromic ones?",
     "Distinct Subsequences", "Dynamic Programming", "comparison", "video"),
    ("eval_082", "How does the wildcard matching DP work?",
     "Wildcard Matching", "Dynamic Programming", "algorithm", "video"),

    # ===================== 10. Recursion / Maths / Complexity =====================
    ("eval_083", "Explain recursion with the recursion tree and stack space.",
     "Introduction to Recursion | Recursion Tree", "Recursion",
     "exact_topic", "video"),
    ("eval_084", "What is the time and space complexity of an algorithm?",
     "Time and Space Complexity", "Complexity", "exact_topic", "video"),
    ("eval_085", "How does merge sort work with its dry run?",
     "Merge Sort | Algorithm | Pseudocode", "Sorting", "algorithm", "video"),
    ("eval_086", "Explain quick sort for beginners.",
     "Quick Sort For Beginners", "Sorting", "algorithm", "video"),
    ("eval_087", "What is the sieve of Eratosthenes?",
     "Sieve of Eratosthenes", "Maths", "algorithm", "video"),
    ("eval_088", "How do I check if a number is prime?",
     "Check if a Number if Prime", "Maths", "algorithm", "video"),
    ("eval_089", "Explain the Euclidean algorithm for GCD.",
     "Basic Maths for DSA | Euclidean Algorithm", "Maths", "algorithm", "video"),
    ("eval_090", "What is hashing and how do maps work internally?",
     "Hashing | Maps | Time Complexity", "Hashing", "concept", "video"),

    # ===================== 11. Greedy =====================
    ("eval_091", "Explain the n meetings in one room problem.",
     "N Meeting in One Room", "Greedy", "algorithm", "video"),
    ("eval_092", "How does the minimum platforms problem work?",
     "Minimum number of platforms", "Greedy", "algorithm", "video"),
    ("eval_093", "What is the fractional knapsack approach?",
     "Fractional Knapsack", "Greedy", "algorithm", "video"),
    ("eval_094", "Explain the jump game problem.",
     "Jump Game - II", "Greedy", "algorithm", "video"),

    # ===================== 12. Terminology / ambiguity probes =====================
    ("eval_095", "What is the difference between a map and an unordered_map in C++ STL?",
     "Complete C++ STL", "STL", "terminology", "video"),
    ("eval_096", "How do I find the middle of a linked list?",
     "middle element of the LinkedList", "Linked List", "ambiguous",
     "video"),
    ("eval_097", "When should I use sliding window over two pointers?",
     "Introduction to Sliding Window and 2 Pointers", "Sliding Window",
     "ambiguous", "video"),
    ("eval_098", "Explain binary search and also its time complexity.",
     "Binary Search Introduction", "Binary Search", "compound", "video"),
    ("eval_099", "How does lower bound work and where is it used in binary search?",
     "Lower Bound and Upper Bound", "Binary Search", "compound", "video"),
    ("eval_100", "What is the approach to find Kth missing positive number?",
     "Kth Missing Positive Number", "Binary Search", "algorithm", "video"),
]

# Chunk-level hints for cases where chunk-precise ground truth is worth
# capturing: eval_id -> list of regexes matched (case-insensitive) against
# transcript chunk text. Matching chunk indexes become relevant_chunk_ids.
# A resolution covering more than CHUNK_MAX_COVERAGE of the video's
# embedded chunks carries no chunk-level information: the case is
# downgraded to video-level instead of pretending to have chunk GT.
CHUNK_HINTS = {
    "eval_021": [r"kadane"],
    "eval_059": [r"breadth[- ]first|\bbfs\b"],
    "eval_060": [r"depth[- ]first|\bdfs\b"],
    "eval_084": [r"big o|big-o|time complexity|space complexity"],
    "eval_085": [r"merge sort|merge step"],
    "eval_073": [r"knapsack"],
}

CHUNK_MAX_COVERAGE = 0.6


def load_metadata_index():
    """video_id -> metadata dict for all videos."""
    index = {}
    for filename in sorted(os.listdir(METADATA_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(METADATA_DIR, filename), encoding="utf-8") as f:
            meta = json.load(f)
        index[meta["video_id"]] = meta
    return index


def resolve_video(title_substring, metadata_index):
    """Resolve a unique title substring to its video id (fail loudly)."""
    matches = [
        (video_id, meta)
        for video_id, meta in metadata_index.items()
        if title_substring.lower() in meta["video_title"].lower()
    ]
    if not matches:
        raise SystemExit(
            f"FATAL: title substring {title_substring!r} matched no video"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"FATAL: title substring {title_substring!r} matched "
            f"{len(matches)} videos: "
            + ", ".join(f"{v} ({m['video_title']})" for v, m in matches)
        )
    return matches[0][0], matches[0][1]["video_title"]


def load_transcript(video_id):
    path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_embedding_chunk_indexes(video_id):
    path = os.path.join(EMBEDDINGS_DIR, f"{video_id}_embeddings.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["chunk_index"] for r in records if isinstance(r.get("chunk_index"), int)}


def resolve_chunks(video_id, patterns):
    """Chunk indexes whose transcript text matches ANY of the patterns.

    Returns (matching_indexes, reason_downgraded). Only indexes that also
    exist in the embedding file are accepted; a transcript chunk that was
    never embedded cannot be retrieved, so it is not valid ground truth.
    """
    transcript = load_transcript(video_id)
    embedded = load_embedding_chunk_indexes(video_id)
    matching = []
    for chunk in transcript:
        text = chunk.get("text", "")
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            index = None
            # chunk_index = position within the transcript chunk list
            # (this is how generate_embeddings numbered them).
            for i, candidate in enumerate(transcript):
                if candidate is chunk:
                    index = i
                    break
            if index is not None and index in embedded:
                matching.append(index)
    return sorted(set(matching))


def build():
    metadata_index = load_metadata_index()
    out_lines = []
    downgrades = []
    chunk_resolutions = {}

    for (eval_id, question, title_key, section, qtype, certainty) in CATALOG:
        video_id, title = resolve_video(title_key, metadata_index)

        chunk_ids = []
        resolved_certainty = certainty
        if eval_id in CHUNK_HINTS and certainty == "video":
            # Try to upgrade to chunk-level ground truth.
            matches = resolve_chunks(video_id, CHUNK_HINTS[eval_id])
            embedded_count = len(load_embedding_chunk_indexes(video_id))
            coverage = (
                len(matches) / embedded_count if embedded_count else 0.0
            )
            if matches and coverage <= CHUNK_MAX_COVERAGE:
                chunk_resolutions[eval_id] = matches
                chunk_ids = [f"{video_id}_{i}" for i in matches]
                resolved_certainty = "exact"
            else:
                downgrades.append((eval_id, video_id, round(coverage, 2)))

        record = {
            "id": eval_id,
            "question": question,
            "relevant_video_ids": [video_id],
            "section": section,
            "question_type": qtype,
            "chunk_certainty": resolved_certainty,
            "notes": f"Ground truth resolved from metadata title: \"{title}\"",
        }
        if chunk_ids:
            record["relevant_chunk_ids"] = chunk_ids
        out_lines.append(json.dumps(record, ensure_ascii=False))

    # Validate the assembled dataset before writing (fail loudly).
    from eval.dataset import load_metadata_video_ids, load_embedding_chunk_keys
    from eval.dataset import validate_dataset

    parsed = [json.loads(line) for line in out_lines]
    normalized = validate_dataset(
        parsed,
        known_video_ids=load_metadata_video_ids(),
        known_chunk_keys=load_embedding_chunk_keys(),
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"Wrote {len(out_lines)} golden cases -> {OUT_PATH}")
    print(f"Chunk-level cases: "
          f"{sum(1 for c in normalized if c['chunk_certainty'] == 'exact')}")
    print(f"Video-level cases: "
          f"{sum(1 for c in normalized if c['chunk_certainty'] == 'video')}")
    if downgrades:
        print(f"Downgraded (no matching chunk): {downgrades}")
    print(f"Chunk resolutions: {chunk_resolutions}")


if __name__ == "__main__":
    build()
