import json

import pytest

from idrkd.evaluation.c3 import C3Case, load_c3_cases, run_c3_benchmark


def test_load_c3_cases_requires_expected_answers_and_unique_ids(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "c3.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "c3-1",
                "query": "Which service handles billing?",
                "expected_answer": "Customer API",
                "tenant_id": "tenant-a",
                "repo_id": "repo-a",
                "expected_entity_ids": ["entity-a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_c3_cases(path) == [
        C3Case(
            case_id="c3-1",
            query="Which service handles billing?",
            expected_answer="Customer API",
            tenant_id="tenant-a",
            repo_id="repo-a",
            expected_entity_ids=("entity-a",),
        )
    ]


async def test_c3_benchmark_is_case_paired_and_bootstraps_deltas() -> None:
    cases = [
        C3Case(
            case_id=f"c3-{index}",
            query="Which service handles billing?",
            expected_answer="Customer API",
            tenant_id="tenant-a",
            repo_id="repo-a",
            expected_entity_ids=("entity-a",),
        )
        for index in range(4)
    ]

    async def monolithic(case: C3Case):  # noqa: ANN202
        return {
            "answer": "Unknown",
            "accepted": False,
            "faithfulness_score": 0.2,
            "retrieved_entity_ids": [],
            "trace": ["monolithic"],
            "case_id": case.case_id,
        }

    async def decomposed(case: C3Case):  # noqa: ANN202
        return {
            "answer": "Customer API",
            "accepted": True,
            "faithfulness_score": 0.9,
            "retrieved_entity_ids": ["entity-a"],
            "trace": ["langgraph", "a2a", "autogen"],
            "case_id": case.case_id,
        }

    artifact = await run_c3_benchmark(
        cases,
        monolithic_runner=monolithic,
        decomposed_runner=decomposed,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert artifact["case_count"] == 4
    assert artifact["error_count"] == 0
    assert artifact["metrics"]["exact_match"]["delta"] == 1.0
    assert artifact["metrics"]["task_completion"]["confidence_interval_95"] == [1.0, 1.0]
    assert artifact["c3_criterion"]["met"] is True
    assert artifact["cases"][0]["monolithic"]["retrieval_recall"] == 0.0
    assert artifact["cases"][0]["langgraph_autogen"]["retrieval_recall"] == 1.0


async def test_c3_benchmark_records_runner_errors_without_claiming_success() -> None:
    case = C3Case(
        case_id="c3-error",
        query="Question",
        expected_answer="Answer",
        tenant_id="tenant-a",
        repo_id="repo-a",
    )

    async def succeeds(_case: C3Case):  # noqa: ANN202
        return {"answer": "Answer", "accepted": True, "faithfulness_score": 1.0}

    async def fails(_case: C3Case):  # noqa: ANN202
        raise RuntimeError("service unavailable")

    artifact = await run_c3_benchmark(
        [case],
        monolithic_runner=succeeds,
        decomposed_runner=fails,
        bootstrap_samples=10,
    )

    assert artifact["error_count"] == 1
    assert artifact["c3_criterion"]["met"] is False
    assert "service unavailable" in artifact["cases"][0]["langgraph_autogen"]["error"]


def test_c3_benchmark_rejects_an_empty_case_set() -> None:
    async def runner(_case: C3Case):  # noqa: ANN202
        return {}

    with pytest.raises(ValueError, match="at least one case"):
        import asyncio

        asyncio.run(run_c3_benchmark([], monolithic_runner=runner, decomposed_runner=runner))
