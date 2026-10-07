# final_test

- Split: test (n = 200)
- Created: 2026-10-05T15:02:50+00:00
- Engine: webiks-hebrew-ragbot Engine.search_documents; model `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0`; index `kolzchut_full*`; 50 candidate paragraphs; top 10 documents
- Title-aware: TitleAwareEngine.search_documents, title weight 0.85
- Code: 4cb2abd

| System | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency median (s) | latency p95 (s) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.360 | 0.575 | 0.650 | 0.755 | 0.488 | 0.519 | 0.664 | 1.053 |
| title_aware | 0.485 | 0.695 | 0.730 | 0.805 | 0.593 | 0.614 | 0.653 | 0.956 |

## title_aware vs baseline

| Metric | Difference | 95% CI |
|---|---|---|
| hit@1 | +0.125 | [+0.060, +0.190] |
| hit@3 | +0.120 | [+0.070, +0.170] |
| hit@5 | +0.080 | [+0.035, +0.125] |
| hit@10 | +0.050 | [+0.005, +0.095] |
| mrr@10 | +0.105 | [+0.061, +0.150] |
| ndcg@10 | +0.095 | [+0.060, +0.131] |

- First relevant rank: improved 73, worsened 25, same 102
- Exact McNemar on hit@3: only baseline 2, only title_aware 26, p = 0.0000

Per-question rows: in the JSON report (question ids, no text).
