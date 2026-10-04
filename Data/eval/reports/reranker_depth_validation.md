# ALGOFORGE — Phase 3B: Reranker Depth Validation

Generated: 2026-09-27T08:15:37+00:00

- Model: `BAAI/bge-reranker-base` — loaded once, reused
- Device: **cuda** | batch size: 16
- Depths: [60, 70, 80]
- Dataset: 100 golden questions

## Candidate-pool invariance proof

PASS — the fused candidate pool (point ids, order) captured pre-rerank is identical across the baseline and every tested depth for all 100 questions. Only the reordering differs; candidate generation is untouched.

## Quality + latency comparison

| Experiment | Depth | Hit@1 | Hit@3 | Hit@5 | Hit@6 | Recall@6 | MRR | Retr p50 | Rerank p50 | Rerank p95 | Total p50 | Total p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | - | 62.0% | 80.0% | 87.0% | 91.0% | 91.0% | 0.725 | 182 ms | - | - | 182 ms | 212 ms |
| CrossEncoder | 60 | 84.0% | 95.0% | 97.0% | 99.0% | 99.0% | 0.900 | 170 ms | 848 ms | 903 ms | 1019 ms | 1090 ms |
| CrossEncoder | 70 | 85.0% | 96.0% | 98.0% | 100.0% | 100.0% | 0.909 | 170 ms | 997 ms | 1069 ms | 1166 ms | 1247 ms |
| CrossEncoder | 80 | 85.0% | 96.0% | 98.0% | 100.0% | 100.0% | 0.909 | 162 ms | 1051 ms | 1194 ms | 1214 ms | 1358 ms |

## Chunk-level case (eval_085, reported separately)

- `eval_085` — pool point-rank 2; baseline rank 2; depth 60: rank 1 (score 0.993); depth 70: rank 1 (score 0.993); depth 80: rank 1 (score 0.993).

NOTE: with only 2 chunk-level golden cases, NO chunk-level precision claim is made — this case is shown for traceability only.

## Video-level failure recovery across depths

| Eval | Expected | Pool point-rank | Baseline | Depth 60 | Depth 70 | Depth 80 |
|---|---|---|---|---|---|---|
| `eval_002` | `6zhGS79oQ4k` | 48 | miss | rank 2 (0.997) | rank 2 (0.997) | rank 2 (0.997) |
| `eval_005` | `qyfekrNni90` | 63 | miss | miss | rank 1 (0.965) | rank 1 (0.965) |
| `eval_012` | `-zSxTJkcdAo` | 41 | miss | rank 1 (0.994) | rank 1 (0.994) | rank 1 (0.994) |
| `eval_018` | `WJaij9ffOIY` | 42 | miss | rank 1 (0.988) | rank 1 (0.988) | rank 1 (0.988) |
| `eval_033` | `wiOo4DC5GGA` | 60 | miss | rank 1 (0.339) | rank 1 (0.339) | rank 1 (0.339) |
| `eval_045` | `jmy0LaGET1I` | 19 | miss | rank 2 (0.976) | rank 2 (0.976) | rank 2 (0.976) |
| `eval_050` | `80Zug6D1_r4` | 47 | miss | rank 1 (0.868) | rank 1 (0.868) | rank 1 (0.868) |
| `eval_057` | `f-sj7I5oXEI` | 51 | miss | rank 1 (0.990) | rank 1 (0.990) | rank 1 (0.990) |
| `eval_099` | `6zhGS79oQ4k` | 50 | miss | rank 2 (0.977) | rank 2 (0.977) | rank 2 (0.977) |

## Determinism

- Two full runs: rankings/metrics identical = **True** (timestamps and wall-clock latency excluded, per spec).
- Reranker scores bit-identical within numerical precision: **True**.

## Production impact

- main.py unchanged: **True** · /ask imports reranker: **False** · Qdrant points: **4850** · golden.jsonl sha256[:16]: `7906f5d4156fef20`

## Limitations

- Only 2 chunk-level golden cases exist; all headline metrics are VIDEO-level. Chunk-level reranking precision is NOT validated by this dataset.
- Validation only: production /ask is unchanged and does not use the reranker.
