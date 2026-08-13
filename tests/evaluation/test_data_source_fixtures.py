import json
from pathlib import Path

from idrkd.evaluation import (
    TaskBenchRunner,
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
    load_tasks_jsonl,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_repository_corpus_fixture_has_five_sources() -> None:
    repos = _read_jsonl(Path("eval/corpus/repository_corpus.jsonl"))

    assert len(repos) == 5
    assert sum(1 for repo in repos if repo["availability"] == "local") == 5
    assert all(repo["source_url"].startswith("https://github.com/") for repo in repos)
    assert all(repo["snapshot_ref"] for repo in repos)
    assert all(repo["local_path"] for repo in repos)


def test_synthetic_schema_fixture_has_conflict_corpus() -> None:
    schemas = _read_jsonl(Path("eval/synthetic_schemas/schemas.jsonl"))
    conflicts = _read_jsonl(Path("eval/synthetic_schemas/conflicts.jsonl"))

    assert len(schemas) == 8
    assert len(conflicts) == 24
    assert {conflict["conflict_type"] for conflict in conflicts} == {
        "missing_required",
        "name_mismatch",
        "type_mismatch",
    }
    assert all(conflict["expected_tool_sequence"] == ["reconcile", "get_conflict", "resolve_conflict"] for conflict in conflicts)


def test_synthetic_schema_fixtures_execute_through_mcp_taskbench() -> None:
    corpus = load_synthetic_schema_corpus()
    tasks = build_synthetic_schema_tasks(corpus)
    registry = build_synthetic_schema_registry(corpus)

    summary = TaskBenchRunner(registry).run(tasks)

    assert len(tasks) == 56
    assert "tenant_id default" in tasks[0].prompt
    assert "repo_id synthetic-schemas" in tasks[0].prompt
    assert summary.pass_rate == 1.0
    assert summary.schema_valid_rate == 1.0
    assert summary.tool_f1 == 1.0
    assert summary.by_category() == {"conflict_resolution": 1.0, "schema_conformance": 1.0}


def test_mcp_task_suite_fixture_has_360_tasks_across_six_categories() -> None:
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))

    assert len(tasks) == 360
    assert {task.category for task in tasks} == {
        "a2a_delegation",
        "conflict_resolution",
        "drift_trigger",
        "multi_hop_planning",
        "schema_conformance",
        "tool_selection",
    }
    assert "Task scope and identifiers as JSON:" in tasks[0].prompt
    assert '"repo_id": "repo-a"' in tasks[0].prompt
    assert '"tenant_id": "default"' in tasks[0].prompt
