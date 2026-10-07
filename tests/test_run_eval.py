import json
from unittest.mock import patch

import pytest

from evaluation import run_eval
from evaluation.engine_setup import DEMO_ROOT
from evaluation.eval_set import Question

ENVIRONMENT = {"versions": {}, "demo_commit": "abc1234", "demo_dirty": False}

QUESTIONS = [
    Question("dev-000", "שאלה א", (1,)),
    Question("dev-001", "שאלה ב", (5, 9)),
    Question("dev-002", "שאלה ג", (42,)),
]


def fake_search(rankings):
    def search(query, top_k):
        return rankings[query][:top_k]

    return search


BASELINE = fake_search({"שאלה א": [3, 1, 2], "שאלה ב": [9, 5, 7], "שאלה ג": [1, 2]})
IMPROVED = fake_search({"שאלה א": [1, 3, 2], "שאלה ב": [9, 5, 7], "שאלה ג": [1, 2, 42]})


def report_for(split, results):
    return run_eval.build_report(
        "test_report",
        split,
        {"splits": {}},
        results,
        top_k=10,
        index_prefix="kolzchut_full",
        model="Fake_Model_v1",
        environment=ENVIRONMENT,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--split", "test", "--name", "x"],
        ["--split", "dev", "--name", "x", "--final"],
        ["--split", "test", "--name", "x", "--final", "--limit", "5"],
        ["--split", "dev", "--name", "x", "--top-k", "5"],
        ["--split", "test", "--name", "x", "--final", "--sweep"],
        ["--split", "heldout", "--name", "x", "--sweep"],
        ["--split", "dev", "--name", "x", "--sweep", "--final"],
        ["--split", "dev", "--name", "x", "--sweep", "--title-weight", "0.2"],
        ["--split", "dev", "--name", "x", "--sweep", "--systems", "title_aware"],
        ["--split", "dev", "--name", "x", "--systems", "title_aware"],
        ["--split", "dev", "--name", "x", "--title-weight", "0.2"],
        ["--split", "dev", "--name", "x", "--systems", "title_aware", "--title-weight", "-0.1"],
        ["--split", "dev", "--name", "x", "--systems", "title_aware", "--title-weight", "nan"],
        ["--split", "dev", "--name", "x", "--systems", "baseline", "baseline"],
        ["--split", "dev", "--name", "x", "--systems", "other"],
        ["--split", "heldout", "--name", "x", "--systems", "title_aware", "--title-weight", "0.2"],
    ],
)
def test_invalid_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        run_eval.parse_args(argv)


def test_valid_title_aware_and_sweep_arguments():
    both = run_eval.parse_args(
        ["--split", "dev", "--name", "x", "--systems", "baseline", "title_aware"]
        + ["--title-weight", "0.25"]
    )
    sweep = run_eval.parse_args(["--split", "dev", "--name", "x", "--sweep"])

    assert both.systems == ["baseline", "title_aware"]
    assert both.title_weight == 0.25
    assert sweep.sweep is True
    assert sweep.systems == ["baseline"]
    assert run_eval.parse_args(["--split", "dev", "--name", "x"]).systems == ["baseline"]


def test_baseline_is_always_the_reference_system():
    args = run_eval.parse_args(
        ["--split", "dev", "--name", "x", "--systems", "title_aware", "baseline"]
        + ["--title-weight", "0.2"]
    )

    assert args.systems == ["baseline", "title_aware"]


def test_sweep_on_test_is_refused_before_loading_anything():
    with (
        patch.object(run_eval, "configure_engine") as configure,
        patch.object(run_eval, "build_eval_set") as build,
        pytest.raises(SystemExit),
    ):
        run_eval.main(["--split", "test", "--final", "--sweep", "--name", "x"])

    configure.assert_not_called()
    build.assert_not_called()


def test_final_test_run_is_allowed_once(tmp_path):
    argv = ["--split", "test", "--name", "final_test", "--final", "--out-dir", str(tmp_path)]

    assert run_eval.parse_args(argv).final is True
    (tmp_path / "final_test.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_eval.parse_args(argv)


def test_test_split_is_refused_before_loading_anything():
    with (
        patch.object(run_eval, "configure_engine") as configure,
        patch.object(run_eval, "build_eval_set") as build,
        pytest.raises(SystemExit),
    ):
        run_eval.main(["--split", "test", "--name", "x"])

    configure.assert_not_called()
    build.assert_not_called()


def test_evaluate_scores_each_question():
    rows = run_eval.evaluate(BASELINE, QUESTIONS, top_k=10)

    assert [r["qid"] for r in rows] == ["dev-000", "dev-001", "dev-002"]
    assert [r["first_relevant_rank"] for r in rows] == [2, 1, None]
    assert rows[0]["retrieved"] == [3, 1, 2]
    assert rows[1]["gold"] == [5, 9]
    assert rows[1]["ndcg@10"] == pytest.approx(1.0)
    assert all(r["latency_s"] >= 0 for r in rows)


def test_summarize_counts_short_rankings():
    summary = run_eval.summarize(run_eval.evaluate(BASELINE, QUESTIONS, top_k=10), top_k=10)

    assert summary["metrics"]["hit@1"] == pytest.approx(1 / 3)
    assert summary["metrics"]["hit@3"] == pytest.approx(2 / 3)
    assert summary["metrics"]["mrr@10"] == pytest.approx(0.5)
    assert summary["short_rankings"] == 3
    assert set(summary["latency_s"]) == {"median", "p95"}


def test_compare_two_systems():
    rows_a = run_eval.evaluate(BASELINE, QUESTIONS, top_k=10)
    rows_b = run_eval.evaluate(IMPROVED, QUESTIONS, top_k=10)

    result = run_eval.compare(rows_a, rows_b)

    assert result["win_loss"] == {"improved": 2, "worsened": 0, "same": 1}
    assert result["mcnemar"] == {"metric": "hit@3", "only_a": 0, "only_b": 1, "p_value": 1.0}
    assert result["differences"]["hit@1"]["diff"] == pytest.approx(1 / 3)


def test_compare_refuses_different_questions():
    rows_a = run_eval.evaluate(BASELINE, QUESTIONS, top_k=10)
    rows_b = run_eval.evaluate(BASELINE, QUESTIONS[:2], top_k=10)

    with pytest.raises(ValueError):
        run_eval.compare(rows_a, rows_b)


def test_heldout_report_has_aggregates_only():
    report = report_for("heldout", {"baseline": run_eval.evaluate(BASELINE, QUESTIONS, 10)})

    assert report["questions"] is None
    assert report["n"] == 3
    assert "comparison" not in report
    assert "none." in run_eval.render_markdown(report)


def test_dev_report_has_rows_with_ids_and_no_text_or_paths(tmp_path):
    results = {
        "baseline": run_eval.evaluate(BASELINE, QUESTIONS, 10),
        "improved": run_eval.evaluate(IMPROVED, QUESTIONS, 10),
    }
    report = report_for("dev", results)

    path = run_eval.write_report(report, tmp_path)

    text = path.read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["questions"][0]["qid"] == "dev-000"
    assert saved["questions"][0]["baseline"]["retrieved"] == [3, 1, 2]
    assert saved["questions"][0]["improved"]["first_relevant_rank"] == 1
    assert saved["comparison"]["systems"] == ["baseline", "improved"]
    assert "שאלה" not in text
    assert str(DEMO_ROOT) not in text
    markdown = (tmp_path / "test_report.md").read_text(encoding="utf-8")
    assert "| baseline |" in markdown
    assert "## improved vs baseline" in markdown


def test_evaluate_systems_alternates_which_system_runs_first():
    calls = []

    def recording(name, search):
        def run(query, top_k):
            calls.append(name)
            return search(query, top_k)

        return run

    results = run_eval.evaluate_systems(
        {"baseline": recording("a", BASELINE), "improved": recording("b", IMPROVED)},
        QUESTIONS,
        top_k=10,
    )

    assert calls == ["a", "b", "b", "a", "a", "b"]
    for system, search in (("baseline", BASELINE), ("improved", IMPROVED)):
        expected = run_eval.evaluate(search, QUESTIONS, 10)
        strip = [{k: v for k, v in row.items() if k != "latency_s"} for row in expected]
        assert [{k: v for k, v in r.items() if k != "latency_s"} for r in results[system]] == strip


def test_report_records_the_title_weight():
    results = {
        "baseline": run_eval.evaluate(BASELINE, QUESTIONS, 10),
        "title_aware": run_eval.evaluate(IMPROVED, QUESTIONS, 10),
    }
    report = run_eval.build_report(
        "final",
        "dev",
        {},
        results,
        top_k=10,
        index_prefix="kolzchut_full",
        model="Fake_Model_v1",
        environment=ENVIRONMENT,
        title_weight=0.125,
    )

    assert report["setup"]["title_weight"] == 0.125
    assert set(report["setup"]["systems"]) == {"baseline", "title_aware"}
    assert report["comparison"]["systems"] == ["baseline", "title_aware"]
    assert "title weight 0.125" in run_eval.render_markdown(report)


def test_weight_grid_is_the_preregistered_one():
    grid = run_eval.WEIGHT_GRID

    assert len(grid) == 21
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert grid[7] == 0.35 and grid[13] == 0.65
    assert all(b > a for a, b in zip(grid, grid[1:], strict=False))


def test_select_weight_takes_the_highest_mrr():
    aggregates = {0.0: {"mrr@10": 0.50}, 0.1: {"mrr@10": 0.55}, 0.2: {"mrr@10": 0.60}}

    assert run_eval.select_weight(aggregates) == 0.2


def test_select_weight_ties_go_to_the_smaller_weight():
    aggregates = {
        0.0: {"mrr@10": 0.50},
        0.1: {"mrr@10": 0.60},
        0.2: {"mrr@10": 0.60 + 1e-15},
        0.3: {"mrr@10": 0.55},
    }

    assert run_eval.select_weight(aggregates) == 0.1
    assert run_eval.select_weight({0.0: {"mrr@10": 0.5}, 0.5: {"mrr@10": 0.5}}) == 0.0


@pytest.mark.parametrize(
    ("selected", "passed"),
    [
        ({"mrr@10": 0.55, "hit@3": 0.60}, True),
        ({"mrr@10": 0.55, "hit@3": 0.59}, True),
        ({"mrr@10": 0.50, "hit@3": 0.70}, False),
        ({"mrr@10": 0.55, "hit@3": 0.58}, False),
    ],
)
def test_gate(selected, passed):
    gate = run_eval.apply_gate({"mrr@10": 0.50, "hit@3": 0.59}, selected, True)

    assert gate["passed"] is passed
    assert gate["mrr@10_higher"] is (selected["mrr@10"] > 0.50)


def test_gate_fails_when_weight_zero_does_not_reproduce_the_engine():
    gate = run_eval.apply_gate({"mrr@10": 0.5, "hit@3": 0.5}, {"mrr@10": 0.6, "hit@3": 0.6}, False)

    assert gate == {
        "w0_matches_engine": False,
        "mrr@10_higher": True,
        "hit@3_not_lower": True,
        "passed": False,
    }


# Fake candidates: the "hits" are the engine's ranking; a positive weight reverses it.
CANDIDATES = [
    (QUESTIONS[0], [3, 1, 2], None),
    (QUESTIONS[1], [9, 5, 7], None),
    (QUESTIONS[2], [1, 2, 42], None),
]


def fake_rank(hits, vector, weight):
    return list(hits) if weight == 0 else list(reversed(hits))


def test_sweep_rows_scores_every_weight():
    rows = run_eval.sweep_rows(CANDIDATES, (0.0, 0.5), fake_rank, top_k=10)

    assert [r["first_relevant_rank"] for r in rows[0.0]] == [2, 1, 3]
    assert [r["first_relevant_rank"] for r in rows[0.5]] == [2, 2, 1]
    assert rows[0.5][2]["retrieved"] == [42, 2, 1]
    assert "latency_s" not in rows[0.0][0]


def test_sweep_rows_keep_top_k():
    rows = run_eval.sweep_rows(CANDIDATES, (0.0,), fake_rank, top_k=2)

    assert rows[0.0][2]["retrieved"] == [1, 2]


def test_sweep_needs_weight_zero():
    with pytest.raises(ValueError):
        run_eval.sweep_rows(CANDIDATES, (0.1, 0.2), fake_rank, top_k=10)


def test_match_engine_counts_identical_rankings():
    rows = run_eval.sweep_rows(CANDIDATES, (0.0,), fake_rank, top_k=10)[0.0]

    assert run_eval.match_engine(rows, [[3, 1, 2], [9, 5, 7], [1, 2, 42]]) == {
        "n": 3,
        "identical": 3,
        "mismatched": [],
    }
    assert run_eval.match_engine(rows, [[3, 1, 2], [5, 9, 7], [1, 2, 42]])["mismatched"] == [
        "dev-001"
    ]


class FakeEngine:
    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query
        self.retrieval_model = self
        self.elastic_model = self

    def encode(self, text):
        return f"vector:{text}"

    def search(self, vector):
        return self.hits_by_query[vector.removeprefix("vector:")]

    def search_documents(self, query, top_k):
        return [{"doc_id": d} for d in self.hits_by_query[query][:top_k]]


def test_collect_candidates_returns_hits_and_engine_rankings():
    engine = FakeEngine({q.text: hits for q, hits, _ in CANDIDATES})

    candidates, engine_rankings = run_eval.collect_candidates(engine, QUESTIONS, top_k=2)

    assert [(q.qid, hits, vector) for q, hits, vector in candidates] == [
        ("dev-000", [3, 1, 2], "vector:שאלה א"),
        ("dev-001", [9, 5, 7], "vector:שאלה ב"),
        ("dev-002", [1, 2, 42], "vector:שאלה ג"),
    ]
    assert engine_rankings == [[3, 1], [9, 5], [1, 2]]


def sweep_report(engine_rankings=([3, 1, 2], [9, 5, 7], [1, 2, 42])):
    rows = run_eval.sweep_rows(CANDIDATES, (0.0, 0.5, 1.0), fake_rank, top_k=10)
    return run_eval.build_sweep_report(
        "sweep",
        {"splits": {}},
        rows,
        run_eval.match_engine(rows[0.0], list(engine_rankings)),
        top_k=10,
        index_prefix="kolzchut_full",
        model="Fake_Model_v1",
        environment=ENVIRONMENT,
    )


def test_sweep_report_selects_and_gates():
    report = sweep_report()

    # MRR@10 is 0.611 at weight 0 and 0.667 at 0.5 and 1.0 (equal), so 0.5 is selected.
    assert report["selection"]["weight"] == 0.5
    assert report["selection"]["on_grid_edge"] is False
    assert report["selection"]["at_zero"]["mrr@10"] == pytest.approx((1 / 2 + 1 + 1 / 3) / 3)
    assert report["selection"]["gate"] == {
        "w0_matches_engine": True,
        "mrr@10_higher": True,
        "hit@3_not_lower": True,
        "passed": True,
    }
    assert [w["weight"] for w in report["weights"]] == [0.0, 0.5, 1.0]
    assert report["comparison"]["systems"] == ["w=0.00", "w=0.50"]
    assert report["comparison"]["win_loss"] == {"improved": 1, "worsened": 1, "same": 1}
    assert report["w0_matches_engine"]["identical"] == 3
    assert report["questions"][2] == {
        "qid": "dev-002",
        "gold": [42],
        "first_relevant_rank": {"0.00": 3, "0.50": 1, "1.00": 1},
        "retrieved": {"0.00": [1, 2, 42], "0.50": [42, 2, 1]},
    }


def test_sweep_report_has_no_text_or_paths(tmp_path):
    path = run_eval.write_report(sweep_report(), tmp_path, run_eval.render_sweep_markdown)

    text = path.read_text(encoding="utf-8")
    markdown = (tmp_path / "sweep.md").read_text(encoding="utf-8")
    assert "שאלה" not in text and "שאלה" not in markdown
    assert str(DEMO_ROOT) not in text
    assert "| Title weight |" in markdown
    assert "| 0.50 |" in markdown
    assert "Selected weight: 0.50" in markdown
    assert ": passed" in markdown
    assert "3/3 questions" in markdown
    assert "## w=0.50 vs w=0.00 (dev)" in markdown


def test_sweep_report_fails_the_gate_when_weight_zero_differs(tmp_path):
    report = sweep_report(engine_rankings=([3, 1, 2], [5, 9, 7], [1, 2, 42]))

    assert report["w0_matches_engine"]["mismatched"] == ["dev-001"]
    assert report["selection"]["gate"]["passed"] is False
    markdown = run_eval.render_sweep_markdown(report)
    assert "failed (weight 0 doesn't reproduce the engine)" in markdown
    assert "2/3 questions" in markdown
