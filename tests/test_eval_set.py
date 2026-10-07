import csv
import hashlib

import pandas as pd
import pytest

from evaluation.eval_set import (
    build_eval_set,
    check_against_ranking_csv,
    gold_doc_ids,
    heldout_questions,
    questions_digest,
    split_dev_test,
)


@pytest.fixture
def qa_df():
    rows = []
    for i in range(40):
        rows.append({"question": f"שאלה {i}", "paragraph": f"p{i}", "link": "l", "doc_id": 100 + i})
        if i % 5 == 0:
            rows.append({"question": f"שאלה {i}", "paragraph": f"x{i}", "link": "l", "doc_id": 7.0})
    return pd.DataFrame(rows)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_heldout_questions_split_unique_questions(qa_df):
    heldout = heldout_questions(qa_df, fraction=0.1, seed=42)

    assert len(heldout) == 4
    assert len(set(heldout)) == 4
    assert set(heldout) <= set(qa_df["question"])


def test_heldout_questions_ignore_duplicate_rows(qa_df):
    deduplicated = qa_df.drop_duplicates("question")

    assert heldout_questions(qa_df) == heldout_questions(deduplicated)


def test_heldout_questions_are_deterministic(qa_df):
    assert heldout_questions(qa_df, seed=42) == heldout_questions(qa_df, seed=42)
    assert heldout_questions(qa_df, seed=42) != heldout_questions(qa_df, seed=1)


def test_split_dev_test_partitions_heldout():
    heldout = [f"q{i}" for i in range(10)]

    dev, test = split_dev_test(heldout, test_size=6, seed=42)

    assert len(dev) == 4
    assert len(test) == 6
    assert set(dev).isdisjoint(test)
    assert sorted(dev + test) == sorted(heldout)
    assert split_dev_test(heldout, test_size=6, seed=42) == (dev, test)


def test_gold_doc_ids_are_sorted_distinct_ints(qa_df):
    gold = gold_doc_ids(qa_df)

    assert gold["שאלה 0"] == (7, 100)
    assert gold["שאלה 1"] == (101,)
    assert all(isinstance(d, int) for d in gold["שאלה 0"])


def test_ranking_check_passes_on_identical_queries(tmp_path):
    path = tmp_path / "ranking_results.csv"
    write_csv(path, ["query", "page 0"], [["שאלה א", "l1"], ["שאלה ב", "l2"]])

    result = check_against_ranking_csv(["שאלה ב", "שאלה א"], path)

    assert result == {
        "file": "ranking_results.csv",
        "rows": 2,
        "matched": 2,
        "missing": 0,
        "extra": 0,
        "ok": True,
    }


def test_ranking_check_reports_missing_and_extra(tmp_path):
    path = tmp_path / "ranking_results.csv"
    write_csv(path, ["query"], [["שאלה א"], ["שאלה ג"]])

    result = check_against_ranking_csv(["שאלה א", "שאלה ב"], path)

    assert result["ok"] is False
    assert (result["matched"], result["missing"], result["extra"]) == (1, 1, 1)


def test_build_eval_set_ids_gold_and_manifest(tmp_path, qa_df):
    qa_csv = tmp_path / "qa.csv"
    qa_df.to_csv(qa_csv, index=False, encoding="utf-8")

    eval_set = build_eval_set(qa_csv, None, heldout_fraction=0.25, test_size=6, seed=42)

    dev, test, heldout = (eval_set.splits[s] for s in ("dev", "test", "heldout"))
    assert (len(dev), len(test), len(heldout)) == (4, 6, 10)
    assert [q.qid for q in dev] == ["dev-000", "dev-001", "dev-002", "dev-003"]
    assert {q.text for q in dev + test} == {q.text for q in heldout}
    assert all(q.gold for q in heldout)

    manifest = eval_set.manifest
    assert manifest["qa_csv"]["rows"] == 48
    assert manifest["qa_csv"]["unique_questions"] == 40
    assert manifest["heldout"]["train_questions"] == 30
    assert manifest["splits"]["dev"]["questions_sha256"] == questions_digest([q.text for q in dev])
    assert manifest["ranking_check"] is None


def test_questions_digest_depends_on_order():
    expected = hashlib.sha256("א\nב".encode()).hexdigest()

    assert questions_digest(["א", "ב"]) == expected
    assert questions_digest(["ב", "א"]) != expected
