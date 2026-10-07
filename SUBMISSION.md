# Title-aware retrieval for the Kol-Zchut RAG Demo

The pages the engine's dense search finds are re-scored by how similar their titles are to the question. On the 200 held-out test questions, Hit@3 rises from 0.575 to 0.695 (+0.120, 95% CI [+0.070, +0.170]). `IS_TITLE_AWARE_RETRIEVAL=true` switches it on; unset, the engine runs unchanged. Interview slides: [interview-presentation.pptx](interview-presentation.pptx).

## 1. The improvement and why
**Direction:** rank pages rather than context-free paragraphs, scoring each page by its paragraph text and its title. Among the baseline's misses inspected during planning, sibling-page confusions were frequent, and only 9.5% of paragraphs contain their page's title.

**Method** (`app/src/retrieval/`, used by `main.py`):
1. The engine's dense search, unchanged: the top 50 paragraphs, best paragraph per page.
2. Each page *d* is re-scored: cos(*q*, best paragraph of *d*) + 0.85 · cos(*q*, title of *d*).

**Why each part is there:** it only reorders, so weight 0 is the engine's ranking; title vectors are stored at indexing time; the query vector is reused.

**Precedent:** BM25F's field weighting (Robertson et al., 2004); passages with their titles ([DPR](https://arxiv.org/abs/2004.04906), [DHR](https://aclanthology.org/2021.findings-emnlp.19/)).

**Not taken:** hybrid BM25 (no gain in a pilot); cross-encoder reranking (7.3–75.9 s per query on a CPU); fine-tuning (a training project); title-prefixed paragraphs (it changes the trained vectors).

## 2. Metrics and why
**Questions.** The model was fine-tuned on the QA dataset, so the evaluation uses the 296 questions the Trainer held out (`test_size=0.1, random_state=42`), split with the same seed into dev (96) and test (200). Relevance is page-level.

**Hit@3 (primary).** The share of questions with a relevant page in the top 3: the Demo passes the top 3 pages (`num_of_pages=3`) to the LLM, one paragraph each. On all 296 held-out questions, the baseline reproduces the Trainer's published Accuracy@3, 0.581.

**Secondary:** Hit@1/5/10, MRR@10, nDCG@10; paired 95% bootstrap CIs; McNemar on Hit@3.

## 3. Results
**Protocol.** The grid (0.00, 0.05, …, 1.00) and the rule were fixed in advance. The weight was selected on dev only, by MRR@10 (0.518 at 0, a plateau of 0.56–0.59 from 0.15, peak 0.588 at 0.85). The test comparison (n = 200, [`final_test.json`](evaluation/reports/final_test.json)) ran once, with frozen code.

| Metric | Baseline | Title-aware | Difference [95% CI] |
|---|---|---|---|
| **Hit@3** | **0.575** | **0.695** | **+0.120 [+0.070, +0.170]** |
| Hit@1 | 0.360 | 0.485 | +0.125 [+0.060, +0.190] |
| Hit@5 | 0.650 | 0.730 | +0.080 [+0.035, +0.125] |
| Hit@10 | 0.755 | 0.805 | +0.050 [+0.005, +0.095] |
| MRR@10 | 0.488 | 0.593 | +0.105 [+0.061, +0.150] |
| nDCG@10 | 0.519 | 0.614 | +0.095 [+0.060, +0.131] |

- **Exact McNemar on Hit@3:** 2 vs 26 questions, p = 3.0e-6.
- **First relevant rank:** 73 improved, 25 worsened, 102 the same.
- **Latency** (CPU, 4 threads): median 0.664 s for the baseline, 0.653 s for title-aware.

## 4. Examples (test)
The two largest rank gains, then the two largest losses; a miss counts as rank 11.

| Question | Relevant page | Rank | Worse system's #1 |
|---|---|---|---|
| …להעסיק מסתננת במסעדה שלנו? | העסקת מבקשי מקלט (מסתננים) | miss&nbsp;→&nbsp;1 | בנסיבות מסוימות ניתן לשלול… |
| …ביטוח מנהלים כולל אובדן כושר עבודה… | ביטוח נכות (אובדן כושר עבודה) במסגרת הביטוח הפנסיוני | miss&nbsp;→&nbsp;1 | מבוטח רשאי לערער… |
| אני בן 70 עצמאי לא קבלתי שום קצבה… | זכותון קצבת זיקנה; מדריך מיצוי זכויות לאזרחים ותיקים | 4&nbsp;→&nbsp;miss | דמי ביטוח לאומי לעצמאי |
| יש טלפון או ארגון בחינם… עובדים מהשטחים? | העסקת עובדים פלסטינים; קו לעובד | 1&nbsp;→&nbsp;7 | שירותי בריאות תעסוקתית לעובדים פלסטינים |

Sources: [Kol-Zchut QA dataset](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-QA-Training-DataSet) (CC BY 4.0); [Kol-Zchut](https://www.kolzchut.org.il/) titles (CC BY-NC-SA 2.5).

## 5. Running locally
Set up as in the [README](README.md#installation): Python 3.10, the model, Elasticsearch, the corpus.
1. Copy `app/.env-example` to `app/.env`; set `ES_EMBEDDING_INDEX=kolzchut_paragraphs`, `IS_MOCK_GPT_CLIENT=true`, and optionally `LOG_LEVEL=INFO` to log the mode.
2. Optional: a 200-paragraph subset (about 11 minutes to index on a laptop CPU, against a day for the corpus); set `PATH_TO_ES_INITIAL_VALUES=../../subset200.json`.
   ```
   python -c "import pandas as pd; pd.read_json('Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json').head(200).to_json('subset200.json')"
   ```
3. From `app/src`, run `python -m uvicorn main:app --port 5000`. Open http://localhost:5000/docs, run `GET /initialize_elastic_from_json`, then `POST /search` with this request body:
   ```
   {"query": "איך עושים אבחון לילד עם הפרעת קשב וריכוז?", "asked_from": "local"}
   ```
   In a POSIX shell (Linux, macOS), curl works too:
   ```
   curl http://localhost:5000/initialize_elastic_from_json
   curl -X POST http://localhost:5000/search -H "Content-Type: application/json" -d '{"query": "איך עושים אבחון לילד עם הפרעת קשב וריכוז?", "asked_from": "local"}'
   ```

**Tests:** `python -m pytest -q`. `IS_TITLE_AWARE_RETRIEVAL=true` is preset; re-index after switching it on. **Full corpus:** index GPU embeddings with `evaluation/index_corpus.py --index-prefix kolzchut_full`, not `/initialize_elastic_from_json`, and set `ES_EMBEDDING_INDEX=kolzchut_full`.

## 6. Reproducing the evaluation
Every step is in [`evaluation/README.md`](evaluation/README.md). The test comparison: `python -m evaluation.run_eval --split test --final --systems baseline title_aware --title-weight 0.85 --name final_test`.

## 7. Limitations
- Relevance is single-label and page-level.
- The 296 held-out questions also chose the model's best training epoch.
- The latency figures are CPU-only, from one machine.
- Inherited from the Demo: a blocking `/search`, updates that delete at most 10 old paragraphs per page, content vectors in the mock answer.
