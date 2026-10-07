# baseline_dev

- Split: dev (n = 96)
- Created: 2026-10-05T13:48:35+00:00
- Engine: webiks-hebrew-ragbot Engine.search_documents; model `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0`; index `kolzchut_full*`; 50 candidate paragraphs; top 10 documents
- Code: e677605

| System | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency median (s) | latency p95 (s) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.396 | 0.594 | 0.656 | 0.771 | 0.518 | 0.524 | 0.720 | 0.919 |

Per-question rows: in the JSON report (question ids, no text).
