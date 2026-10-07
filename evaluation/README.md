# Retrieval evaluation

Tools that measure how well the Demo's search finds the right Kol-Zchut page for a question.

| Script | Purpose |
|---|---|
| `eval_set.py` | Rebuilds the held-out questions and splits them into dev and test |
| `embed_corpus.py` | Embeds every paragraph and page title with the retrieval model (GPU or CPU, resumable) |
| `index_corpus.py` | Indexes the full corpus in Elasticsearch from those embeddings, in the engine's document shape |
| `run_eval.py` | Runs the questions through the engine's search, or the title-aware search, and writes a report; selects the title weight on dev |
| `metrics.py` | Hit@k, MRR@10, nDCG@10, win/loss counts, paired bootstrap CIs, exact McNemar test |
| `engine_setup.py` | Shared paths, and the environment the engine reads at import time |

Commands run from the repository root, with the Demo's virtual environment active (Python 3.10).

## Evaluation questions
The retrieval model was fine-tuned on the QA dataset, so only questions it didn't train on are fair to evaluate on.
- **Held out (296).** The Trainer's `split_train_eval` holds out 10% of the unique questions: `train_test_split(unique_questions, test_size=0.1, random_state=42)`.
- **Dev (96) / test (200).** `train_test_split(heldout, test_size=200, random_state=42)` splits the held-out questions. Dev is used for selecting the weight; the test comparison runs once, at the end.
- **Relevance.** Document level: a page is relevant to a question if a QA row pairs them.

## Metrics
All metrics are at the document level, over the top 10 documents the search returns.
- **Hit@3 (primary).** The share of questions with a relevant page among the top 3: the 3 pages whose paragraph goes to the LLM (`num_of_pages=3`). The Trainer reports it as Accuracy@3.
- **Hit@1, Hit@5 and Hit@10.** The same measure at other cutoffs.
- **MRR@10.** The mean reciprocal rank of the first relevant page; 0 if none is in the top 10.
- **nDCG@10.** Binary relevance and a log2 discount, the BEIR/MTEB standard.
- **Comparisons.** A run with two systems also reports:
  - the paired difference of every metric, with a 95% bootstrap CI (10,000 resamples, seed 0);
  - an exact McNemar test on Hit@3;
  - win/loss counts of first-relevant ranks (a miss counts as rank 11).

## Data
The data files aren't part of this repository: the Kol-Zchut content is licensed CC BY-NC-SA 2.5. By default, the scripts look for the data where the Demo does; every path can be passed as an option.

| File | Default location | Source |
|---|---|---|
| Paragraph corpus `Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json` | repository root | [Drive](https://drive.google.com/file/d/1w4d6O5pnFAVPExb9SzcaCXRL8qjT9blB/view), [about](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-Paragraph-Corpus) |
| QA dataset `Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv` | repository root | [Drive](https://drive.google.com/file/d/1YaQ8ZbpqfBzZvxZZQv01tlSSnVnFBgMz/view), [about](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-QA-Training-DataSet) |
| Retrieval model `Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0/` | `app/artifacts/` | [Drive (zip)](https://drive.google.com/file/d/1i_7bTdGWC7yUVC_NLDQGk63kPRhZT7y3/view) |
| Embeddings `paragraphs.npy`, `titles.npy`, `manifest.json` | any folder | `embed_corpus.py` (below) |

## 1. Evaluation set
```
python -m evaluation.eval_set --qa-csv PATH/TO/Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv
```
This prints a manifest: counts, seeds and a SHA-256 for each ordered question list.
- Expected: 3,890 rows; 2,951 unique questions; 296 held out (96 dev, 200 test).
- When the model directory holds the Trainer's `ranking_results.csv`, the held-out questions are also checked against its queries (296/296).

Reports identify questions by split and position (`dev-017` is the 18th dev question) and contain no dataset text. To read a question:
```python
from evaluation.eval_set import build_eval_set
dev = {q.qid: q.text for q in build_eval_set("PATH/TO/QA.csv").splits["dev"]}
```

## 2. Embeddings
`embed_corpus.py` embeds the 24,487 paragraphs and the 7,007 distinct page titles with the released model.
- **Cost.** A few seconds per paragraph on a laptop CPU (about a day for the corpus); minutes on a GPU.
- **Shards.** It writes fixed-size shards, each one atomically, and a rerun skips the shards that are already complete.
- **Finalize.** `--finalize` merges the shards and writes `manifest.json`: model and corpus SHA-256, counts, order hashes, library versions and device.

### On Google Colab (GPU)
About half an hour on a T4. Colab runs Python 3.13, where the Demo's sentence-transformers 3.0.1 and transformers 4.42.3 can't be installed (tokenizers fails to build). The cells therefore install current versions, unpinned. The verification step below re-embeds a sample with the Demo's versions and catches any numerical drift.

1. Open a new notebook and choose Runtime → Change runtime type → T4 GPU.
2. Run the cells below in order. Cell 1 asks for a file: choose `evaluation/embed_corpus.py`.
3. At the end, the browser downloads `paragraphs.npy`, `titles.npy` and `manifest.json`; put them into one local folder.

Nothing is written to Drive, so nothing survives a disconnect: a disconnected run restarts from the beginning. For a run that resumes, use the Drive variant below.

Cell 1 — libraries and script:
```python
import os
from google.colab import files
WORK = "/content/hebrew-rag"
MODEL = "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"
CORPUS = "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
!pip install -q sentence-transformers gdown
!nvidia-smi --query-gpu=name,memory.total --format=csv
files.upload()
```

Cell 2 — corpus and model, from the official links:
```python
!gdown 1w4d6O5pnFAVPExb9SzcaCXRL8qjT9blB -O "{CORPUS}"
!gdown 1i_7bTdGWC7yUVC_NLDQGk63kPRhZT7y3 -O "{MODEL}.zip"
!unzip -q -o "{MODEL}.zip" -d models && ls "models/{MODEL}"
```

Cell 3 — embed, merge, download:
```python
!python embed_corpus.py --corpus "{CORPUS}" --model-dir "models/{MODEL}" --out-dir embeddings --batch-size 32
!python embed_corpus.py --finalize --corpus "{CORPUS}" --model-dir "models/{MODEL}" --out-dir embeddings
for name in ("paragraphs.npy", "titles.npy", "manifest.json"):
    files.download(f"embeddings/{name}")
```

**Drive variant (resumable).** The shards are written to Google Drive, so a rerun after a disconnect skips the completed ones.
1. In Google Drive, create `MyDrive/hebrew-rag/` and upload `evaluation/embed_corpus.py` into it.
2. Replace cell 1 with:
   ```python
   import os
   from google.colab import drive
   drive.mount("/content/drive")
   WORK = "/content/drive/MyDrive/hebrew-rag"
   MODEL = "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0"
   CORPUS = "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"
   os.makedirs("/content/inputs", exist_ok=True)
   os.chdir("/content/inputs")
   !pip install -q sentence-transformers gdown
   ```
3. Run cell 2 unchanged; it keeps the corpus and model on the runtime's local disk.
4. Replace cell 3 with:
   ```python
   !python "{WORK}/embed_corpus.py" --corpus "{CORPUS}" --model-dir "models/{MODEL}" --out-dir "{WORK}/embeddings" --batch-size 32
   !python "{WORK}/embed_corpus.py" --finalize --corpus "{CORPUS}" --model-dir "models/{MODEL}" --out-dir "{WORK}/embeddings"
   ```
5. After a disconnect, reconnect and run all cells again.
6. When the run is done, download `paragraphs.npy`, `titles.npy` and `manifest.json` from `MyDrive/hebrew-rag/embeddings/` (not the `shards` folder).

Troubleshooting:
- **GPU out of memory:** use `--batch-size 16`.
- **`gdown` download quota:** upload the corpus JSON and the model zip yourself. Without Drive, use `files.upload()` in the working folder; with Drive, upload them to `MyDrive/hebrew-rag/` and copy them to `/content/inputs`.

### On the local CPU (fallback)
The same script runs locally; leave it running in the background, and rerun the same command to resume after an interruption:
```
python evaluation/embed_corpus.py --corpus CORPUS.json --model-dir app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0 --out-dir EMBEDDINGS_DIR --device cpu --threads 8
python evaluation/embed_corpus.py --finalize --corpus CORPUS.json --model-dir app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0 --out-dir EMBEDDINGS_DIR
```

### Verify
```
python evaluation/embed_corpus.py --verify-sample 20 --corpus CORPUS.json --model-dir app/artifacts/Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0 --out-dir EMBEDDINGS_DIR
```
This checks:
- the corpus and model SHA-256 and the order hashes against the local files;
- shapes, finite values and unit norms.

It then re-embeds 20 paragraphs and 10 titles on this machine; every cosine with the stored vectors must be at least 0.9999. It exits with code 1 on any failure.

## 3. Index the full corpus
Start Elasticsearch (see the main README), then run:
```
python -m evaluation.index_corpus --embeddings-dir EMBEDDINGS_DIR --corpus CORPUS.json --index-prefix kolzchut_full --recreate
```
- **Documents.** It writes one document per paragraph, as the engine's `create_paragraphs` would: same fields, `index_from_doc_id` index names, dynamic mapping. Each document also carries its page's title vector. A document's `_id` is its corpus position, so a rerun overwrites instead of duplicating.
- **Verification.** It checks:
  - every document is there (24,487 in 23 indices `kolzchut_full_0` … `kolzchut_full_22`);
  - both vector fields exist on all of them and are mapped as `dense_vector`;
  - 20 sampled documents hold exactly the vectors from the embedding files.

  `--verify-only` reruns the checks without indexing.
- **Prefix.** `--recreate` deletes only the indices named `<prefix>_<number>`. Prefixes must not be prefixes of each other, because `/initialize_elastic_from_json` deletes every index matching `ES_EMBEDDING_INDEX*`. The full corpus uses `kolzchut_full`, so the backend's `ES_EMBEDDING_INDEX` must be a different name that isn't a prefix of it, e.g. `kolzchut_paragraphs`.

## 4. Evaluate
```
python -m evaluation.run_eval --split dev --name baseline_dev --qa-csv QA.csv
python -m evaluation.run_eval --split heldout --name baseline_sanity_heldout296 --qa-csv QA.csv
python -m evaluation.run_eval --split test --final --systems baseline title_aware --title-weight W --name final_test --qa-csv QA.csv
```
- **Search.** It loads the engine in-process, as the backend does, and sends each question through a system's `search_documents`:
  - `baseline` (the default): `Engine.search_documents` — the query embedding, the top 50 paragraphs by cosine similarity, then the best paragraph per document, keeping 10 documents.
  - `title_aware`: `TitleAwareEngine.search_documents` (`app/src/retrieval/`) — the same candidates, each document re-scored as `cos(query, best paragraph) + W · cos(query, title)`. It needs `--title-weight W` and an index with title vectors (section 3).
- **Two systems.** With `--systems baseline title_aware`, both run on every question and the report compares title-aware with the baseline: the difference of every metric with its bootstrap CI, the McNemar test on Hit@3 and the win/loss counts.
- **Latency.** Each question's latency covers the whole call, including the query embedding. With two systems, the one that runs first alternates from question to question.
- **Reports.** They go to `evaluation/reports/<name>.json` and `.md`.
  - Dev and final test reports have one row per question: the question id, the relevant and retrieved doc_ids, the rank and the metrics.
  - `heldout` (all 296 questions, including test) writes aggregates only. It serves as a sanity check against the numbers published with the model, without exposing per-question test outcomes.
- **Test split.** It needs `--final`, and the run refuses to overwrite an existing report of the same name. `title_aware` doesn't run on `heldout`, which contains the test questions.
- **Configuration.** `--index-prefix`, `--es-url` and `--model-location` select another index, server or model copy. `--limit N` runs the first N questions, for smoke tests.

## 5. Title weight (dev only)
```
python -m evaluation.run_eval --split dev --sweep --name weight_sweep_dev --qa-csv QA.csv
```
- **Sweep.** Each dev question's embedding and its 50 candidate paragraphs are computed once. They are then re-ranked for every weight w in 0.00, 0.05, …, 1.00 with the same ranking function the title-aware engine uses.
- **Check.** The engine's own `Engine.search_documents` also runs once per question, and weight 0 must reproduce its ranking on every question. Otherwise the gate fails and the script exits with code 1.
- **Selection.** The weight with the highest dev MRR@10; ties go to the smaller weight.
- **Gate.** The selected weight is used only if the check passed, its dev MRR@10 is higher than at weight 0, and its Hit@3 isn't lower.
- **Report.** Metrics for every weight, the selection, the gate, and the comparison of the selected weight with weight 0 on dev.

The grid and the rules are fixed in advance. The weight is selected on dev only; the comparison on the test split ran once.

## Tests
```
python -m pytest -q tests/test_eval_set.py tests/test_embed_corpus.py tests/test_index_corpus.py tests/test_metrics.py tests/test_run_eval.py tests/test_title_aware_engine.py tests/test_retrieval_settings.py
```
The tests use small synthetic data, a fake encoder and a fake Elasticsearch client; they need neither the model nor a running Elasticsearch.
