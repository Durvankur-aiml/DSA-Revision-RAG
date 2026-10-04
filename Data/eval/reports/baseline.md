# ALGOFORGE — Retrieval Baseline Report

Generated: 2026-09-27T05:50:46+00:00  
Git commit: `178d849`  
Dataset: `D:\Mission - Anthropic\DSA RAG\Data\eval\golden.jsonl` (100 questions)

## 1. Retrieval configuration under test

- TOP_K: **6**
- Fusion weights: title **0.6** · semantic **0.3** · text **0.1**
- Exact-title boost: **+0.2** when title_score >= 0.95
- MIN_SCORE_THRESHOLD: **0.25** (semantic) / 0.70 (title)
- Candidate pools: semantic top-40 + lexical top-40
- Embeddings: all-MiniLM-L6-v2 (384d, cosine)
- Qdrant collection: `striver_a2z`

## 2. Dataset coverage

- 100 questions: 98 video-level · 2 chunk-level
- 16 distinct sections: Arrays, BST, Binary Search, Binary Trees, Complexity, Dynamic Programming, Graphs, Greedy, Hashing, Linked List, Maths, Recursion, STL, Sliding Window, Sorting, Stack and Queue
- Question types: algorithm, ambiguous, comparison, complexity, compound, concept, exact_topic, implementation, terminology

## 3. Aggregate metrics

### Final retrieval (production top-6 selection)

| metric | value | n |
|---|---|---|
| Final Hit@1 | 62.0% | 100 |
| Final Hit@3 | 79.0% | 100 |
| Final Hit@5 | 86.0% | 100 |
| Final Hit@6 | 90.0% | 100 |
| Final Hit@10 | 90.0% | 100 |
| Final Recall@6 | 89.3% | 100 |
| Final Recall@20 | 89.3% | 100 |
| Final MRR | 0.720 | 100 |

### Candidate retrieval (semantic 40 ∪ lexical 40, pre-selection)

| metric | value | n |
|---|---|---|
| Candidate Hit@1 | 62.0% | 100 |
| Candidate Hit@3 | 79.0% | 100 |
| Candidate Hit@5 | 87.0% | 100 |
| Candidate Hit@6 | 91.0% | 100 |
| Candidate Hit@10 | 92.0% | 100 |
| Candidate Recall@20 | 97.3% | 100 |
| Candidate MRR | 0.728 | 100 |

### Ground-truth levels (final Hit@6, never mixed)

| ground truth | Hit@1 | Hit@3 | Hit@6 | n |
|---|---|---|---|---|
| video-level | 63.3% | 79.6% | 90.8% | 98 |
| chunk-level | 0.0% | 50.0% | 50.0% | 2 |

### Health indicators

- Average final source count: 6.00
- No-result rate: 0.0%
- Exact-topic candidate rate: 0.0%
- Retrieval latency: p50 84.0 ms · p95 99.3 ms · mean 102.7 ms (retrieval only — no Gemini)

## 4. Failure categories

| category | count | share of failures |
|---|---|---|
| `beyond_top_k` | 9 | 90% |
| `correct_video_wrong_chunk` | 1 | 10% |

## 5. Failed questions (all)

### `eval_002` — beyond_top_k

> What is lower bound and upper bound in binary search?

- Expected video(s): `6zhGS79oQ4k`
- Retrieved: `Bsv3FPUX_BA`, `uZ0N_hZpyps`, `F9c7LpRZWVQ`, `UvBKTVaG6U8`, `Z0hwjftStI4`, `kMSBvlZ-_HA`
- Best candidate rank: 15
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_005` — beyond_top_k

> How does binary search work on the answer space, like Koko eating bananas?

- Expected video(s): `qyfekrNni90`
- Retrieved: `uZ0N_hZpyps`, `MHf6awe89xw`, `Bsv3FPUX_BA`, `Z0hwjftStI4`, `kMSBvlZ-_HA`, `UvBKTVaG6U8`
- Best candidate rank: 21
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_012` — beyond_top_k

> How does the longest substring without repeating characters work?

- Expected video(s): `-zSxTJkcdAo`
- Retrieved: `p7-9UvDQZ3w`, `xm_W1ub-K-w`, `kouxiP_H5WE`, `X0oXMdtUDwo`, `KcNt6v_56cc`, `UmJT3j26t1I`
- Best candidate rank: 14
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_018` — beyond_top_k

> What is the minimum window substring approach?

- Expected video(s): `WJaij9ffOIY`
- Retrieved: `X0oXMdtUDwo`, `9TJYWh0adfk`, `kouxiP_H5WE`, `p7-9UvDQZ3w`, `KcNt6v_56cc`, `UmJT3j26t1I`
- Best candidate rank: 14
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_033` — beyond_top_k

> How does Floyd's cycle detection work for linked lists?

- Expected video(s): `wiOo4DC5GGA`
- Retrieved: `sWf7k1x9XR4`, `qf6qp7GzD5Q`, `8ocB7a_c-Cc`, `YbY8cVwWAvw`, `2Kd0KKmmHFc`, `WAOfKpxYHR8`
- Best candidate rank: 19
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_045` — beyond_top_k

> Explain the tree traversals: inorder, preorder and postorder.

- Expected video(s): `jmy0LaGET1I`
- Retrieved: `aZNaLrVebKQ`, `9GMECGQgWrQ`, `Bfqd8BsPVuw`, `LgLRTaEMRVc`, `ySp2epYvgTE`, `BhuvF_-PWS0`
- Best candidate rank: 10
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_050` — beyond_top_k

> Explain Morris traversal for inorder without recursion or stack.

- Expected video(s): `80Zug6D1_r4`
- Retrieved: `yVdKa8dnKiE`, `kvRjNm4rVBE`, `-uQGzhYj8BQ`, `nGJmxkUJQGs`, `GqOmJHQZivw`, `69ZCDFy-OUo`
- Best candidate rank: 16
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_057` — beyond_top_k

> How do I validate whether a binary tree is a BST?

- Expected video(s): `f-sj7I5oXEI`
- Retrieved: `X0oXMdtUDwo`, `Yt50Jfbd8Po`, `0ca1nvR0be4`, `u-yWemKGWO0`, `Rezetez59Nk`, `9GMECGQgWrQ`
- Best candidate rank: 25
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

### `eval_085` — correct_video_wrong_chunk

> How does merge sort work with its dry run?

- Expected video(s): `ogjf7ORKfd8`
- Expected chunk(s): `ogjf7ORKfd8_0`, `ogjf7ORKfd8_12`, `ogjf7ORKfd8_33`
- Retrieved: `8ocB7a_c-Cc`, `ogjf7ORKfd8`, `WIrA4YexLRQ`, `FMwpt_aQOGw`, `1zktEppsdig`, `0e4bZaP3MDI`
- Best candidate rank: 2
- Detail: expected video selected but the labeled chunk is not

### `eval_099` — beyond_top_k

> How does lower bound work and where is it used in binary search?

- Expected video(s): `6zhGS79oQ4k`
- Retrieved: `Bsv3FPUX_BA`, `MHf6awe89xw`, `uZ0N_hZpyps`, `F9c7LpRZWVQ`, `kMSBvlZ-_HA`, `UvBKTVaG6U8`
- Best candidate rank: 16
- Detail: relevant chunks survived thresholding but lost the top-k race to other videos

## 6. Per-question results

| id | level | hit@6 | cand@20 recall | MRR | failure |
|---|---|---|---|---|---|
| `eval_001` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_002` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_003` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_004` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_005` | video | 0.0% | 0.0% | 0.00 | beyond_top_k |
| `eval_006` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_007` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_008` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_009` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_010` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_011` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_012` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_013` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_014` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_015` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_016` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_017` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_018` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_019` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_020` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_021` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_022` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_023` | video | 100.0% | 100.0% | 0.17 | — |
| `eval_024` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_025` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_026` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_027` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_028` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_029` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_030` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_031` | video | 100.0% | 100.0% | 0.17 | — |
| `eval_032` | video | 100.0% | 100.0% | 0.17 | — |
| `eval_033` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_034` | video | 100.0% | 100.0% | 0.20 | — |
| `eval_035` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_036` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_037` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_038` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_039` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_040` | video | 100.0% | 100.0% | 0.33 | — |
| `eval_041` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_042` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_043` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_044` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_045` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_046` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_047` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_048` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_049` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_050` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_051` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_052` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_053` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_054` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_055` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_056` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_057` | video | 0.0% | 0.0% | 0.00 | beyond_top_k |
| `eval_058` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_059` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_060` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_061` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_062` | video | 100.0% | 100.0% | 0.33 | — |
| `eval_063` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_064` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_065` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_066` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_067` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_068` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_069` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_070` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_071` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_072` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_073` | exact | 100.0% | 100.0% | 0.33 | — |
| `eval_074` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_075` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_076` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_077` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_078` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_079` | video | 100.0% | 100.0% | 0.33 | — |
| `eval_080` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_081` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_082` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_083` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_084` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_085` | exact | 0.0% | 33.3% | 0.00 | correct_video_wrong_chunk |
| `eval_086` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_087` | video | 100.0% | 100.0% | 0.33 | — |
| `eval_088` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_089` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_090` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_091` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_092` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_093` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_094` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_095` | video | 100.0% | 100.0% | 0.50 | — |
| `eval_096` | video | 100.0% | 100.0% | 0.17 | — |
| `eval_097` | video | 100.0% | 100.0% | 1.00 | — |
| `eval_098` | video | 100.0% | 100.0% | 0.25 | — |
| `eval_099` | video | 0.0% | 100.0% | 0.00 | beyond_top_k |
| `eval_100` | video | 100.0% | 100.0% | 1.00 | — |
