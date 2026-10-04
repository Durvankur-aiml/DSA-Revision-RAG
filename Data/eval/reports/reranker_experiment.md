# ALGOFORGE — Phase 3: Cross-Encoder Reranking Experiment

Generated: 2026-09-27T06:53:21+00:00

- Model: `BAAI/bge-reranker-base` (loaded once, reused)
- Device: **cuda** | batch size: 16
- Dataset: 100 golden questions (`D:\Mission - Anthropic\DSA RAG\Data\eval\golden.jsonl`)
- Qdrant collection: `striver_a2z` (read-only)
- Git commit: `178d849`

## Comparison

| Experiment | Depth | Hit@1 | Hit@3 | Hit@5 | Hit@6 | Recall@6 | Recall@20(cand) | MRR | Retr p50 | Rerank p50 | Rerank p95 | Total p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | - | 62.0% | 80.0% | 87.0% | 91.0% | 91.0% | 98.0% | 0.725 | 171 ms | - | - | 171 ms |
| CrossEncoder | 10 | 73.0% | 81.0% | 87.0% | 90.0% | 90.0% | 98.0% | 0.787 | 168 ms | 137 ms | 156 ms | 304 ms |
| CrossEncoder | 20 | 75.0% | 83.0% | 89.0% | 92.0% | 92.0% | 98.0% | 0.807 | 155 ms | 275 ms | 303 ms | 429 ms |
| CrossEncoder | 30 | 75.0% | 84.0% | 88.0% | 92.0% | 92.0% | 98.0% | 0.808 | 158 ms | 409 ms | 444 ms | 566 ms |
| CrossEncoder | 40 | 74.0% | 84.0% | 88.0% | 92.0% | 92.0% | 98.0% | 0.803 | 134 ms | 560 ms | 607 ms | 688 ms |
| CrossEncoder | 60 | 84.0% | 95.0% | 97.0% | 99.0% | 99.0% | 98.0% | 0.900 | 170 ms | 897 ms | 951 ms | 1069 ms |

## Phase 2 failure recovery

| eval | question | expected | pool rank | baseline final | reranker final | reranker score |
|---|---|---|---|---|---|---|
| `eval_002` | What is lower bound and upper bound in binary search? | `6zhGS79oQ4k` | 48 | miss | 2 | 0.9967621564865112 |
| `eval_005` | How does binary search work on the answer space, like Koko e | `qyfekrNni90` | 63 | miss | miss | - |
| `eval_012` | How does the longest substring without repeating characters  | `-zSxTJkcdAo` | 41 | miss | 1 | 0.9938557744026184 |
| `eval_018` | What is the minimum window substring approach? | `WJaij9ffOIY` | 42 | miss | 1 | 0.9882320761680603 |
| `eval_033` | How does Floyd's cycle detection work for linked lists? | `wiOo4DC5GGA` | 60 | miss | 1 | 0.3385142385959625 |
| `eval_045` | Explain the tree traversals: inorder, preorder and postorder | `jmy0LaGET1I` | 19 | miss | 2 | 0.9757874011993408 |
| `eval_050` | Explain Morris traversal for inorder without recursion or st | `80Zug6D1_r4` | 47 | miss | 1 | 0.8682459592819214 |
| `eval_057` | How do I validate whether a binary tree is a BST? | `f-sj7I5oXEI` | 51 | miss | 1 | 0.9901195764541626 |
| `eval_099` | How does lower bound work and where is it used in binary sea | `6zhGS79oQ4k` | 50 | miss | 2 | 0.97704017162323 |

## Limitations

- Only 2 chunk-level cases exist in the golden dataset; all headline metrics are VIDEO-level and chunk-level reranking quality cannot be assessed from this dataset.
- This is an experiment: production retrieval is unchanged and no reranker is wired into the API path.
