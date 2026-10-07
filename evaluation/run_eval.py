"""Evaluate retrieval on the held-out questions, in-process, with the engine the backend uses.

Each question is sent through a system's ``search_documents`` and the returned documents
are scored against the QA dataset's relevant documents. Systems:
- ``baseline``: the engine's search (query embedding, the Elasticsearch search over the top
  50 paragraphs, best paragraph per document).
- ``title_aware``: the same search, with each document re-scored by
  ``+ weight * cos(query, title)`` (``app/src/retrieval``).

Splits:
- ``dev`` (96): tuning and analysis; the report has one row per question.
- ``heldout`` (296, dev + test): sanity check against the published numbers; the report
  holds aggregates only, so no per-question test outcome is seen before the final run.
- ``test`` (200): the final run only, with ``--final``.

``--sweep`` selects the title weight on dev: each question's embedding and candidates are
computed once and re-ranked for every weight in ``WEIGHT_GRID``.

Usage:
    python -m evaluation.run_eval --split dev --name baseline_dev [--qa-csv PATH]
    python -m evaluation.run_eval --split heldout --name baseline_sanity_heldout296
    python -m evaluation.run_eval --split dev --sweep --name weight_sweep_dev
    python -m evaluation.run_eval --split test --final --systems baseline title_aware \
        --title-weight W --name final_test
"""

import argparse
import datetime
import importlib.metadata
import json
import logging
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.engine_setup import (
    DEFAULT_MODEL_LOCATION,
    DEFAULT_QA_CSV,
    DEFAULT_REPORTS_DIR,
    DEMO_ROOT,
    configure_engine,
    model_name,
)
from evaluation.eval_set import Question, build_eval_set
from evaluation.metrics import (
    CUTOFF,
    METRICS,
    aggregate,
    latency_summary,
    mcnemar_exact,
    metric_values,
    paired_bootstrap_ci,
    question_metrics,
    win_loss,
)

if TYPE_CHECKING:
    # The engine reads its settings when imported, so it is imported only after
    # configure_engine (see main).
    from webiks_hebrew_ragbot.engine import Engine

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[int]]
RankFn = Callable[[list[dict], object, float], list[int]]
Candidate = tuple[Question, list[dict], object]

DEFAULT_INDEX_PREFIX = "kolzchut_full"
CANDIDATE_PARAGRAPHS = 50
ROWS_SPLITS = {"dev", "test"}
PRIMARY_METRIC = "hit@3"
SYSTEMS = ("baseline", "title_aware")
SYSTEM_DESCRIPTIONS = {
    "baseline": "Engine.search_documents (webiks-hebrew-ragbot)",
    "title_aware": "TitleAwareEngine.search_documents: + title_weight * cos(query, title)",
}
# The grid and the selection and gate rules were fixed before any dev run, so the selected
# weight isn't tuned to the rules that judge it.
WEIGHT_GRID = tuple(round(0.05 * i, 2) for i in range(21))
SELECTION_METRIC = "mrr@10"
SELECTION_RULE = "highest mrr@10; ties (within 1e-12) go to the smaller weight"
GATE_RULE = (
    "weight 0 reproduces the engine on every question, "
    "mrr@10(w*) > mrr@10(0) and hit@3(w*) >= hit@3(0)"
)
TIE_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command line.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        argparse.Namespace: The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=("dev", "test", "heldout"), required=True)
    parser.add_argument("--name", required=True, help="report file name, without extension")
    parser.add_argument("--final", action="store_true", help="required for the test split")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["baseline"])
    parser.add_argument("--title-weight", type=float, default=None, help="title_aware weight")
    parser.add_argument("--sweep", action="store_true", help="select the title weight on dev")
    parser.add_argument("--qa-csv", type=Path, default=DEFAULT_QA_CSV)
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX)
    parser.add_argument("--model-location", type=Path, default=DEFAULT_MODEL_LOCATION)
    parser.add_argument("--top-k", type=int, default=CUTOFF)
    parser.add_argument("--limit", type=int, default=None, help="first N questions (smoke runs)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        parser.error("the test split runs once, at the end; pass --final to run it")
    if args.final and args.split != "test":
        parser.error("--final applies to the test split only")
    if args.final and args.limit is not None:
        parser.error("--final runs every test question; drop --limit")
    if args.final and (args.out_dir / f"{args.name}.json").exists():
        parser.error(f"{args.name}.json already exists; the final test run is not repeated")
    if args.top_k < CUTOFF:
        parser.error(f"--top-k must be at least {CUTOFF}, the metrics' cutoff")
    if len(set(args.systems)) != len(args.systems):
        parser.error("list each system once")
    if args.sweep:
        if args.split != "dev":
            parser.error("the title weight is selected on dev only; --sweep needs --split dev")
        if args.systems != ["baseline"] or args.title_weight is not None:
            parser.error("--sweep compares the weights itself; drop --systems and --title-weight")
    elif "title_aware" in args.systems:
        if args.split == "heldout":
            # heldout includes the test questions; with the dev report, its aggregates would
            # reveal title-aware test results before the final run.
            parser.error("the title_aware system runs on dev, or on test with --final")
        if args.title_weight is None:
            parser.error("the title_aware system needs --title-weight")
        if not math.isfinite(args.title_weight) or args.title_weight < 0:
            parser.error("--title-weight must be a number >= 0")
    elif args.title_weight is not None:
        parser.error("--title-weight applies to the title_aware system only")
    # A fixed order keeps the baseline as the reference of every comparison.
    args.systems = [system for system in SYSTEMS if system in args.systems]
    return args


def score_question(question: Question, retrieved: list[int]) -> dict:
    """Score one question's ranking.

    Args:
        question (Question): The question and its relevant documents.
        retrieved (list[int]): Retrieved document ids, best first.

    Returns:
        dict: ``qid``, ``gold``, ``retrieved`` and the metrics.
    """
    return {
        "qid": question.qid,
        "gold": list(question.gold),
        "retrieved": retrieved,
        **question_metrics(retrieved, question.gold),
    }


def evaluate_systems(
    searches: dict[str, SearchFn], questions: Sequence[Question], top_k: int
) -> dict[str, list[dict]]:
    """Run every question through every system and score the rankings.

    The system that runs first rotates from question to question, so that no system's
    latency is measured consistently before or after the other's.

    Args:
        searches (dict[str, SearchFn]): Search function per system name.
        questions (Sequence[Question]): Questions with their relevant documents.
        top_k (int): Documents to retrieve per question.

    Returns:
        dict[str, list[dict]]: Per system, one row per question: id, gold, retrieved,
            metrics and latency.
    """
    names = list(searches)
    rows = {name: [] for name in names}
    for i, question in enumerate(questions, start=1):
        shift = (i - 1) % len(names)
        for name in names[shift:] + names[:shift]:
            start = time.perf_counter()
            retrieved = searches[name](question.text, top_k)
            latency = time.perf_counter() - start
            rows[name].append({**score_question(question, retrieved), "latency_s": latency})
        if i % 25 == 0 or i == len(questions):
            logger.info("Evaluated %d/%d questions", i, len(questions))
    return rows


def evaluate(search: SearchFn, questions: Sequence[Question], top_k: int) -> list[dict]:
    """Run every question through a search function and score the ranking.

    Args:
        search (SearchFn): Returns the ranked document ids for a query and ``top_k``.
        questions (Sequence[Question]): Questions with their relevant documents.
        top_k (int): Documents to retrieve per question.

    Returns:
        list[dict]: One row per question: id, gold, retrieved, metrics and latency.
    """
    return evaluate_systems({"system": search}, questions, top_k)["system"]


def summarize(rows: Sequence[dict], top_k: int) -> dict:
    """Aggregate one system's per-question rows.

    Args:
        rows (Sequence[dict]): Per-question rows.
        top_k (int): Documents requested per question.

    Returns:
        dict: Mean metrics, latency median and p95, and how many rankings were shorter
            than ``top_k`` (the 50 candidate paragraphs covered fewer documents).
    """
    return {
        "metrics": aggregate(rows),
        "latency_s": latency_summary([row["latency_s"] for row in rows]),
        "short_rankings": sum(1 for row in rows if len(row["retrieved"]) < top_k),
    }


def compare(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> dict:
    """Compare two systems on the same questions.

    Args:
        rows_a (Sequence[dict]): Reference system rows.
        rows_b (Sequence[dict]): Compared system rows, same questions in the same order.

    Returns:
        dict: Win/loss counts of first relevant ranks, a paired-bootstrap CI for each
            metric difference (b - a), and the exact McNemar test on the primary metric.
    """
    if [row["qid"] for row in rows_a] != [row["qid"] for row in rows_b]:
        raise ValueError("The systems were evaluated on different questions")
    return {
        "win_loss": win_loss(
            [row["first_relevant_rank"] for row in rows_a],
            [row["first_relevant_rank"] for row in rows_b],
        ),
        "differences": {
            metric: paired_bootstrap_ci(
                metric_values(rows_a, metric), metric_values(rows_b, metric)
            )
            for metric in METRICS
        },
        "mcnemar": {
            "metric": PRIMARY_METRIC,
            **mcnemar_exact(
                metric_values(rows_a, PRIMARY_METRIC), metric_values(rows_b, PRIMARY_METRIC)
            ),
        },
    }


def question_rows(results: dict[str, list[dict]]) -> list[dict]:
    """Merge the systems' rows into one row per question.

    Args:
        results (dict[str, list[dict]]): Rows per system name, same question order.

    Returns:
        list[dict]: ``qid``, ``gold`` and, per system, its retrieved ids, metrics and latency.
    """
    merged = []
    for rows in zip(*results.values(), strict=True):
        entry = {"qid": rows[0]["qid"], "gold": rows[0]["gold"]}
        for system, row in zip(results, rows, strict=True):
            entry[system] = {k: v for k, v in row.items() if k not in ("qid", "gold")}
        merged.append(entry)
    return merged


def created_utc() -> str:
    """Return the current UTC time for report headers.

    Returns:
        str: ISO 8601 time, to the second.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def base_setup(model: str, index_prefix: str, top_k: int) -> dict:
    """Describe the retrieval setup shared by every report.

    Args:
        model (str): Embedding model name.
        index_prefix (str): Elasticsearch index prefix searched.
        top_k (int): Documents retrieved per question.

    Returns:
        dict: The setup.
    """
    return {
        "engine": "webiks-hebrew-ragbot Engine.search_documents",
        "model": model,
        "index_prefix": index_prefix,
        "candidate_paragraphs": CANDIDATE_PARAGRAPHS,
        "top_k": top_k,
        "relevance": "document level: doc_ids paired with the question in the QA dataset",
    }


def build_report(
    name: str,
    split: str,
    eval_manifest: dict,
    results: dict[str, list[dict]],
    *,
    top_k: int,
    index_prefix: str,
    model: str,
    environment: dict,
    limit: int | None = None,
    title_weight: float | None = None,
) -> dict:
    """Assemble the report.

    Per-question rows are included for dev and test only, and contain ids, never text.

    Args:
        name (str): Report name.
        split (str): ``dev``, ``test`` or ``heldout``.
        eval_manifest (dict): Manifest of the evaluation set.
        results (dict[str, list[dict]]): Per-question rows per system name.
        top_k (int): Documents retrieved per question.
        index_prefix (str): Elasticsearch index prefix searched.
        model (str): Embedding model name.
        environment (dict): Library versions and code state.
        limit (int | None): Number of questions when a run was limited.
        title_weight (float | None): Weight of the title-aware system, if it ran.

    Returns:
        dict: The report.
    """
    n = len(next(iter(results.values())))
    setup = base_setup(model, index_prefix, top_k)
    setup["systems"] = {system: SYSTEM_DESCRIPTIONS.get(system) for system in results}
    setup["title_weight"] = title_weight
    report = {
        "name": name,
        "created_utc": created_utc(),
        "split": split,
        "n": n,
        "limit": limit,
        "setup": setup,
        "eval_set": eval_manifest,
        "environment": environment,
        "systems": {system: summarize(rows, top_k) for system, rows in results.items()},
    }
    if len(results) == 2:
        rows_a, rows_b = results.values()
        report["comparison"] = {"systems": list(results), **compare(rows_a, rows_b)}
    report["questions"] = question_rows(results) if split in ROWS_SPLITS else None
    return report


def render_header(report: dict) -> list[str]:
    """Render the Markdown title and the run description.

    Args:
        report (dict): A report.

    Returns:
        list[str]: Markdown lines.
    """
    setup = report["setup"]
    lines = [
        f"# {report['name']}",
        "",
        f"- Split: {report['split']} (n = {report['n']})"
        + (f", limited to the first {report['limit']}" if report["limit"] else ""),
        f"- Created: {report['created_utc']}",
        f"- Engine: {setup['engine']}; model `{setup['model']}`; "
        f"index `{setup['index_prefix']}*`; {setup['candidate_paragraphs']} candidate paragraphs; "
        f"top {setup['top_k']} documents",
    ]
    if setup.get("title_weight") is not None:
        lines.append(
            f"- Title-aware: TitleAwareEngine.search_documents, title weight "
            f"{setup['title_weight']:g}"
        )
    lines.append(
        f"- Code: {report['environment'].get('demo_commit')}"
        + (" (uncommitted changes)" if report["environment"].get("demo_dirty") else "")
    )
    return lines


def render_comparison(comparison: dict, label: str = "") -> list[str]:
    """Render a comparison of two systems as Markdown.

    Args:
        comparison (dict): The ``comparison`` section of a report.
        label (str): Text appended to the section title.

    Returns:
        list[str]: Markdown lines.
    """
    a, b = comparison["systems"]
    wl = comparison["win_loss"]
    mcnemar = comparison["mcnemar"]
    lines = [
        "",
        f"## {b} vs {a}{label}",
        "",
        "| Metric | Difference | 95% CI |",
        "|---|---|---|",
    ]
    for metric, d in comparison["differences"].items():
        lines.append(f"| {metric} | {d['diff']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] |")
    lines += [
        "",
        f"- First relevant rank: improved {wl['improved']}, worsened {wl['worsened']}, "
        f"same {wl['same']}",
        f"- Exact McNemar on {mcnemar['metric']}: only {a} {mcnemar['only_a']}, "
        f"only {b} {mcnemar['only_b']}, p = {mcnemar['p_value']:.4f}",
    ]
    return lines


def render_markdown(report: dict) -> str:
    """Render a short Markdown summary of a report.

    Args:
        report (dict): The report.

    Returns:
        str: Markdown text.
    """
    lines = render_header(report) + [
        "",
        "| System | " + " | ".join(METRICS) + " | latency median (s) | latency p95 (s) |",
        "|---|" + "---|" * (len(METRICS) + 2),
    ]
    for system, summary in report["systems"].items():
        values = " | ".join(f"{summary['metrics'][m]:.3f}" for m in METRICS)
        latency = summary["latency_s"]
        lines.append(f"| {system} | {values} | {latency['median']:.3f} | {latency['p95']:.3f} |")

    if report.get("comparison"):
        lines += render_comparison(report["comparison"])
    lines += [
        "",
        "Per-question rows: "
        + ("in the JSON report (question ids, no text)." if report["questions"] else "none."),
        "",
    ]
    return "\n".join(lines)


def weight_key(weight: float) -> str:
    """Format a title weight as a report key.

    Args:
        weight (float): Title weight.

    Returns:
        str: The weight with two decimals, e.g. ``0.35``.
    """
    return f"{weight:.2f}"


def sweep_rows(
    candidates: Sequence[Candidate], weights: Sequence[float], rank: RankFn, top_k: int
) -> dict[float, list[dict]]:
    """Re-rank each question's candidates for every weight and score the rankings.

    Args:
        candidates (Sequence[Candidate]): Per question: the question, the engine's
            Elasticsearch hits and the query embedding, computed once.
        weights (Sequence[float]): Title weights; must include 0, the engine's ranking.
        rank (RankFn): Ranked document ids for hits, a query embedding and a weight.
        top_k (int): Documents kept per question.

    Returns:
        dict[float, list[dict]]: Per weight, one scored row per question.

    Raises:
        ValueError: If the weights don't include 0.
    """
    if 0.0 not in weights:
        raise ValueError("The weights must include 0, the engine's own ranking")
    return {
        weight: [
            score_question(question, rank(hits, vector, weight)[:top_k])
            for question, hits, vector in candidates
        ]
        for weight in weights
    }


def match_engine(rows: Sequence[dict], engine_rankings: Sequence[list[int]]) -> dict:
    """Check that rankings equal the engine's own, question by question.

    Args:
        rows (Sequence[dict]): Scored rows, e.g. of weight 0.
        engine_rankings (Sequence[list[int]]): The engine's ``search_documents`` ids.

    Returns:
        dict: ``n``, ``identical`` (count) and the ``mismatched`` question ids.
    """
    mismatched = [
        row["qid"]
        for row, ranking in zip(rows, engine_rankings, strict=True)
        if row["retrieved"] != ranking
    ]
    return {"n": len(rows), "identical": len(rows) - len(mismatched), "mismatched": mismatched}


def select_weight(aggregates: dict[float, dict]) -> float:
    """Select the weight with the highest dev MRR@10; ties go to the smaller weight.

    Args:
        aggregates (dict[float, dict]): Aggregate metrics per weight.

    Returns:
        float: The selected weight.
    """
    best = None
    for weight in sorted(aggregates):
        score = aggregates[weight][SELECTION_METRIC]
        if best is None or score > aggregates[best][SELECTION_METRIC] + TIE_TOLERANCE:
            best = weight
    return best


def apply_gate(at_zero: dict, at_selected: dict, w0_matches_engine: bool) -> dict:
    """Apply the pre-registered gate to the selected weight.

    The comparison with weight 0 stands for a comparison with the engine only if weight 0
    reproduces the engine's ranking, so that is a condition too.

    Args:
        at_zero (dict): Aggregate metrics at weight 0 (the engine's ranking).
        at_selected (dict): Aggregate metrics at the selected weight.
        w0_matches_engine (bool): Whether weight 0 reproduced the engine on every question.

    Returns:
        dict: Each condition's outcome and ``passed``.
    """
    checks = {
        "w0_matches_engine": w0_matches_engine,
        "mrr@10_higher": at_selected["mrr@10"] > at_zero["mrr@10"],
        "hit@3_not_lower": at_selected["hit@3"] >= at_zero["hit@3"],
    }
    return {**checks, "passed": all(checks.values())}


def build_sweep_report(
    name: str,
    eval_manifest: dict,
    rows_by_weight: dict[float, list[dict]],
    w0_check: dict,
    *,
    top_k: int,
    index_prefix: str,
    model: str,
    environment: dict,
    limit: int | None = None,
) -> dict:
    """Assemble the dev weight-sweep report: metrics per weight, selection and gate.

    Args:
        name (str): Report name.
        eval_manifest (dict): Manifest of the evaluation set.
        rows_by_weight (dict[float, list[dict]]): Scored dev rows per weight.
        w0_check (dict): ``match_engine`` result for weight 0.
        top_k (int): Documents retrieved per question.
        index_prefix (str): Elasticsearch index prefix searched.
        model (str): Embedding model name.
        environment (dict): Library versions and code state.
        limit (int | None): Number of questions when a run was limited.

    Returns:
        dict: The report.
    """
    weights = sorted(rows_by_weight)
    aggregates = {weight: aggregate(rows_by_weight[weight]) for weight in weights}
    selected = select_weight(aggregates)
    zero_rows, selected_rows = rows_by_weight[0.0], rows_by_weight[selected]
    setup = base_setup(model, index_prefix, top_k)
    setup.update(
        {
            "ranking": "rank_documents: cos(query, best paragraph) + w * cos(query, title)",
            "weights": weights,
            "selection_rule": SELECTION_RULE,
            "gate_rule": GATE_RULE,
        }
    )
    return {
        "name": name,
        "created_utc": created_utc(),
        "split": "dev",
        "n": len(zero_rows),
        "limit": limit,
        "setup": setup,
        "eval_set": eval_manifest,
        "environment": environment,
        "w0_matches_engine": w0_check,
        "weights": [{"weight": weight, "metrics": aggregates[weight]} for weight in weights],
        "selection": {
            "weight": selected,
            "on_grid_edge": selected in (weights[0], weights[-1]),
            "at_zero": aggregates[0.0],
            "at_selected": aggregates[selected],
            "gate": apply_gate(aggregates[0.0], aggregates[selected], not w0_check["mismatched"]),
        },
        "comparison": {
            "systems": [f"w={weight_key(0.0)}", f"w={weight_key(selected)}"],
            **compare(zero_rows, selected_rows),
        },
        "questions": [
            {
                "qid": row["qid"],
                "gold": row["gold"],
                "first_relevant_rank": {
                    weight_key(weight): rows_by_weight[weight][i]["first_relevant_rank"]
                    for weight in weights
                },
                "retrieved": {
                    weight_key(0.0): row["retrieved"],
                    weight_key(selected): selected_rows[i]["retrieved"],
                },
            }
            for i, row in enumerate(zero_rows)
        ],
    }


def render_sweep_markdown(report: dict) -> str:
    """Render a short Markdown summary of a weight-sweep report.

    Args:
        report (dict): The sweep report.

    Returns:
        str: Markdown text.
    """
    check = report["w0_matches_engine"]
    selection = report["selection"]
    gate = selection["gate"]
    at_zero, at_selected = selection["at_zero"], selection["at_selected"]
    lines = render_header(report) + [
        f"- Weight 0 reproduces Engine.search_documents on {check['identical']}/{check['n']} "
        "questions",
        "",
        "| Title weight | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for entry in report["weights"]:
        values = " | ".join(f"{entry['metrics'][m]:.3f}" for m in METRICS)
        lines.append(f"| {weight_key(entry['weight'])} | {values} |")
    lines += [
        "",
        "## Selection and gate",
        "",
        f"- Selected weight: {weight_key(selection['weight'])} "
        f"({report['setup']['selection_rule']})"
        + ("; on the edge of the grid" if selection["on_grid_edge"] else ""),
        f"- Gate ({report['setup']['gate_rule']}): "
        f"mrr@10 {at_zero['mrr@10']:.3f} -> {at_selected['mrr@10']:.3f}, "
        f"hit@3 {at_zero['hit@3']:.3f} -> {at_selected['hit@3']:.3f}: "
        + ("passed" if gate["passed"] else "failed")
        + ("" if gate["w0_matches_engine"] else " (weight 0 doesn't reproduce the engine)"),
    ]
    lines += render_comparison(report["comparison"], label=" (dev)")
    lines += ["", "Per-question rows: in the JSON report (question ids, no text).", ""]
    return "\n".join(lines)


def write_report(
    report: dict, out_dir: Path, render: Callable[[dict], str] = render_markdown
) -> Path:
    """Write the report as JSON and Markdown.

    Args:
        report (dict): The report.
        out_dir (Path): Output directory.
        render (Callable[[dict], str]): Markdown renderer.

    Returns:
        Path: The JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{report['name']}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(out_dir / f"{report['name']}.md", "w", encoding="utf-8") as f:
        f.write(render(report))
    return json_path


def git_state() -> dict:
    """Return the Demo repository's commit and whether tracked files have changes.

    Returns:
        dict: ``demo_commit`` and ``demo_dirty``; None values when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(DEMO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(DEMO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"demo_commit": None, "demo_dirty": None}
    return {"demo_commit": commit, "demo_dirty": bool(status)}


def environment_info(es_version: str | None) -> dict:
    """Collect the versions and code state that the numbers depend on.

    Args:
        es_version (str | None): Elasticsearch server version.

    Returns:
        dict: Versions, torch threads and git state.
    """
    import torch

    packages = ("webiks-hebrew-ragbot", "sentence-transformers", "transformers", "torch")
    versions = {"python": platform.python_version()}
    for package in packages + ("elasticsearch", "numpy"):
        versions[package] = importlib.metadata.version(package)
    return {
        "versions": versions,
        "elasticsearch_server": es_version,
        "torch_threads": torch.get_num_threads(),
        **git_state(),
    }


def engine_search(engine: "Engine") -> SearchFn:
    """Wrap an engine's public search as a function returning document ids.

    Args:
        engine (Engine): A ``webiks_hebrew_ragbot`` Engine, or a subclass.

    Returns:
        SearchFn: The search function.
    """

    def search(query: str, top_k: int) -> list[int]:
        return [int(doc["doc_id"]) for doc in engine.search_documents(query, top_k)]

    return search


def collect_candidates(
    engine: "Engine", questions: Sequence[Question], top_k: int
) -> tuple[list[Candidate], list[list[int]]]:
    """Fetch each question's candidates, and the engine's own ranking for the weight-0 check.

    The candidates come from the calls ``search_documents`` makes (query embedding, then
    the Elasticsearch search), once per question, so every weight re-ranks the same hits.
    The engine's public ``search_documents`` then runs once more per question; its ranking
    is what weight 0 must reproduce.

    Args:
        engine (Engine): The engine whose search provides the candidates.
        questions (Sequence[Question]): Questions.
        top_k (int): Documents for the engine's own ranking.

    Returns:
        tuple[list[Candidate], list[list[int]]]: The candidates per question, and the
            engine's ``search_documents`` ranking per question, for the weight-0 check.
    """
    candidates, engine_rankings = [], []
    search = engine_search(engine)
    for i, question in enumerate(questions, start=1):
        vector = engine.retrieval_model.encode(question.text)
        candidates.append((question, engine.elastic_model.search(vector), vector))
        engine_rankings.append(search(question.text, top_k))
        if i % 25 == 0 or i == len(questions):
            logger.info("Fetched candidates for %d/%d questions", i, len(questions))
    return candidates, engine_rankings


def run_sweep(
    args: argparse.Namespace,
    engine: "Engine",
    questions: Sequence[Question],
    eval_manifest: dict,
    model: str,
    environment: dict,
) -> int:
    """Run the dev weight sweep and write its report.

    Args:
        args (argparse.Namespace): The arguments.
        engine (Engine): The engine whose search provides the candidates.
        questions (Sequence[Question]): Dev questions.
        eval_manifest (dict): Manifest of the evaluation set.
        model (str): Embedding model name.
        environment (dict): Library versions and code state.

    Returns:
        int: Exit code; 1 when weight 0 doesn't reproduce the engine's ranking.
    """
    from retrieval.title_aware_engine import rank_documents, title_vector_field

    title_field = title_vector_field(engine.model_name)

    def rank(hits: list[dict], vector, weight: float) -> list[int]:
        ranked = rank_documents(hits, vector, weight, title_field, engine.identifier_field)
        return [int(document.doc_id) for document in ranked]

    candidates, engine_rankings = collect_candidates(engine, questions, args.top_k)
    rows_by_weight = sweep_rows(candidates, WEIGHT_GRID, rank, args.top_k)
    w0_check = match_engine(rows_by_weight[0.0], engine_rankings)
    report = build_sweep_report(
        args.name,
        eval_manifest,
        rows_by_weight,
        w0_check,
        top_k=args.top_k,
        index_prefix=args.index_prefix,
        model=model,
        environment=environment,
        limit=args.limit,
    )
    path = write_report(report, args.out_dir, render_sweep_markdown)
    selection = report["selection"]
    logger.info(
        "Wrote %s: weight 0 matches the engine on %d/%d; selected %s; gate %s",
        path.name,
        w0_check["identical"],
        w0_check["n"],
        weight_key(selection["weight"]),
        "passed" if selection["gate"]["passed"] else "failed",
    )
    if w0_check["mismatched"]:
        logger.error("Weight 0 differs from the engine on %s", ", ".join(w0_check["mismatched"]))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write the report.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Exit code.
    """
    args = parse_args(argv)
    # The engine logs every search on the root logger; show only this package's progress.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("evaluation", __name__):
        logging.getLogger(name).setLevel(logging.INFO)

    model = model_name()
    ranking_csv = args.model_location / model / "ranking_results.csv"
    eval_set = build_eval_set(args.qa_csv, ranking_csv)
    questions = eval_set.splits[args.split][: args.limit]

    configure_engine(args.index_prefix, args.model_location)
    from elasticsearch import Elasticsearch
    from retrieval.title_aware_engine import TitleAwareEngine
    from webiks_hebrew_ragbot.engine import Engine

    es = Elasticsearch(args.es_url, request_timeout=60)
    engine = Engine(llms_client=None, es_client=es)
    environment = environment_info(es.info()["version"]["number"])
    if args.sweep:
        logger.info(
            "Sweeping %d title weights on %d dev questions", len(WEIGHT_GRID), len(questions)
        )
        return run_sweep(args, engine, questions, eval_set.manifest, model, environment)

    engines = {"baseline": engine}
    if "title_aware" in args.systems:
        engines["title_aware"] = TitleAwareEngine(
            llms_client=None,
            es_client=es,
            retrieval_model=engine.retrieval_model,
            title_weight=args.title_weight,
        )
    searches = {system: engine_search(engines[system]) for system in args.systems}
    for search in searches.values():
        search(questions[0].text, args.top_k)

    logger.info(
        "Evaluating %s on %d %s questions on %s*",
        ", ".join(searches),
        len(questions),
        args.split,
        args.index_prefix,
    )
    results = evaluate_systems(searches, questions, args.top_k)
    report = build_report(
        args.name,
        args.split,
        eval_set.manifest,
        results,
        top_k=args.top_k,
        index_prefix=args.index_prefix,
        model=model,
        environment=environment,
        limit=args.limit,
        title_weight=args.title_weight,
    )
    path = write_report(report, args.out_dir)
    for system, summary in report["systems"].items():
        metrics = summary["metrics"]
        logger.info(
            "Wrote %s, %s: %s",
            path.name,
            system,
            ", ".join(f"{m} {metrics[m]:.3f}" for m in METRICS),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
