from idrkd.evaluation import McpTask


def test_evaluation_package_exports_task_model() -> None:
    task = McpTask(
        id="x",
        category="tool_selection",
        prompt="Find code",
        expected_tool="search_code",
        arguments={"tenant_id": "t", "repo_id": "r", "query": "q"},
    )

    assert task.expected_call().name == "search_code"
