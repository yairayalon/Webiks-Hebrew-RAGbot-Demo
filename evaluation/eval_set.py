"""Held-out evaluation questions: the Trainer's split, divided into dev and test.

The retrieval model was trained on the QA dataset, so only the questions the Trainer held
out are fair to evaluate on. The Trainer's ``split_train_eval`` keeps 10% of the unique
questions (seed 42): 296 questions. They are split once more into dev (96, for tuning and
analysis) and test (200, for one final run). A question's relevant documents are every
``doc_id`` the QA dataset pairs it with.

Usage:
    python -m evaluation.eval_set [--qa-csv PATH] [--ranking-csv PATH]
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from evaluation.engine_setup import DEFAULT_MODEL_LOCATION, DEFAULT_QA_CSV, file_sha256, model_name

logger = logging.getLogger(__name__)

HELDOUT_FRACTION = 0.1
TEST_SIZE = 200
SEED = 42
SPLITS = ("dev", "test", "heldout")


@dataclass(frozen=True)
class Question:
    """An evaluation question.

    Attributes:
        qid (str): Stable id: split name and position in the split, e.g. ``dev-017``.
        text (str): The question.
        gold (tuple[int, ...]): Relevant document ids, sorted.
    """

    qid: str
    text: str
    gold: tuple[int, ...]


@dataclass(frozen=True)
class EvalSet:
    """The evaluation questions by split, with a manifest that identifies them.

    Attributes:
        splits (dict[str, list[Question]]): ``dev``, ``test`` and ``heldout`` (dev + test).
        manifest (dict): Counts, seeds and hashes of the ordered question lists.
    """

    splits: dict[str, list[Question]]
    manifest: dict


def heldout_questions(
    df: pd.DataFrame, fraction: float = HELDOUT_FRACTION, seed: int = SEED
) -> list[str]:
    """Reproduce the Trainer's held-out questions (``utils.split_train_eval``).

    Args:
        df (pd.DataFrame): The QA dataset, with a ``question`` column.
        fraction (float): Share of unique questions held out.
        seed (int): ``random_state`` of the split.

    Returns:
        list[str]: Held-out questions, in the order the split returns them.
    """
    unique_questions = df["question"].unique()
    _, heldout = train_test_split(unique_questions, test_size=fraction, random_state=seed)
    return [str(q) for q in heldout]


def split_dev_test(
    heldout: list[str], test_size: int = TEST_SIZE, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Split the held-out questions into dev and test.

    Args:
        heldout (list[str]): Held-out questions, in split order.
        test_size (int): Number of test questions.
        seed (int): ``random_state`` of the split.

    Returns:
        tuple[list[str], list[str]]: Dev and test questions.
    """
    dev, test = train_test_split(heldout, test_size=test_size, random_state=seed)
    return list(dev), list(test)


def gold_doc_ids(df: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Map each question to the documents the QA dataset pairs it with.

    Args:
        df (pd.DataFrame): The QA dataset, with ``question`` and ``doc_id`` columns.

    Returns:
        dict[str, tuple[int, ...]]: Sorted, distinct document ids per question.
    """
    return {
        str(question): tuple(sorted({int(d) for d in doc_ids}))
        for question, doc_ids in df.groupby("question", sort=False)["doc_id"]
    }


def questions_digest(questions: list[str]) -> str:
    """Hash an ordered question list, so a split can be identified without its text.

    Args:
        questions (list[str]): Questions in split order.

    Returns:
        str: SHA-256 hex digest of the newline-joined UTF-8 text.
    """
    return hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()


def check_against_ranking_csv(heldout: list[str], path: Path) -> dict:
    """Compare the held-out questions with the queries of the model's ``ranking_results.csv``.

    The Trainer's evaluator wrote that file for its held-out questions, so the two sets
    must be identical.

    Args:
        heldout (list[str]): Held-out questions.
        path (Path): The ranking file.

    Returns:
        dict: Row count, matched, missing and extra counts, and ``ok``.
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    queries = {row[0] for row in rows if row}
    expected = set(heldout)
    return {
        "file": path.name,
        "rows": len(rows),
        "matched": len(expected & queries),
        "missing": len(expected - queries),
        "extra": len(queries - expected),
        "ok": queries == expected,
    }


def build_eval_set(
    qa_csv: Path,
    ranking_csv: Path | None = None,
    *,
    heldout_fraction: float = HELDOUT_FRACTION,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> EvalSet:
    """Build the dev, test and held-out questions with their relevant documents.

    Args:
        qa_csv (Path): The QA dataset CSV.
        ranking_csv (Path | None): The model's ``ranking_results.csv`` to check against;
            skipped when None or missing.
        heldout_fraction (float): Share of unique questions held out.
        test_size (int): Number of test questions.
        seed (int): ``random_state`` of both splits.

    Returns:
        EvalSet: Questions by split and the manifest.
    """
    df = pd.read_csv(qa_csv, encoding="utf-8")
    heldout = heldout_questions(df, heldout_fraction, seed)
    dev, test = split_dev_test(heldout, test_size, seed)
    gold = gold_doc_ids(df)
    texts = {"dev": dev, "test": test, "heldout": heldout}
    splits = {
        name: [Question(f"{name}-{i:03d}", q, gold[q]) for i, q in enumerate(questions)]
        for name, questions in texts.items()
    }

    ranking_check = None
    if ranking_csv is not None and Path(ranking_csv).is_file():
        ranking_check = check_against_ranking_csv(heldout, Path(ranking_csv))
    else:
        logger.info("Ranking file not found; skipping the check against it")

    unique_count = int(df["question"].nunique())
    manifest = {
        "qa_csv": {
            "file": Path(qa_csv).name,
            "sha256": file_sha256(Path(qa_csv)),
            "rows": len(df),
            "unique_questions": unique_count,
        },
        "heldout": {
            "rule": "train_test_split(unique_questions, test_size, random_state)[1]",
            "test_size": heldout_fraction,
            "random_state": seed,
            "train_questions": unique_count - len(heldout),
        },
        "dev_test": {
            "rule": "dev, test = train_test_split(heldout, test_size, random_state)",
            "test_size": test_size,
            "random_state": seed,
        },
        "splits": {
            name: {
                "n": len(questions),
                "relevant_pairs": sum(len(gold[q]) for q in questions),
                "questions_sha256": questions_digest(questions),
            }
            for name, questions in texts.items()
        },
        "scikit_learn": sklearn.__version__,
        "ranking_check": ranking_check,
    }
    return EvalSet(splits=splits, manifest=manifest)


def main(argv: list[str] | None = None) -> int:
    """Print the evaluation-set manifest.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Exit code; 1 when the ranking check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--qa-csv", type=Path, default=DEFAULT_QA_CSV)
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=None,
        help="defaults to ranking_results.csv in the model directory under app/artifacts",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ranking_csv = args.ranking_csv or DEFAULT_MODEL_LOCATION / model_name() / "ranking_results.csv"
    eval_set = build_eval_set(args.qa_csv, ranking_csv)
    print(json.dumps(eval_set.manifest, indent=2))
    check = eval_set.manifest["ranking_check"]
    return 1 if check is not None and not check["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
