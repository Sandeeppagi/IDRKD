import os

from idrkd.mcp.live_smoke import LiveSmokeConfig, LiveSmokeResult, config_from_env


def test_live_smoke_result_serializes_expected_fields() -> None:
    result = LiveSmokeResult(
        tenant_id="tenant-live",
        repo_id="week5-e2e",
        tools_count=14,
        search_hits=3,
        bfs_neighbors=4,
        path_hops=2,
        source_id="source",
        target_id="target",
    )

    assert result.as_dict()["tenant_id"] == "tenant-live"
    assert result.as_dict()["path_hops"] == 2


def test_live_smoke_config_prefers_tenant_and_repo_env(monkeypatch) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "TENANT_ID": "tenant-live",
            "REPO_ID": "week5-e2e",
            "MCP_BASE_URL": "http://mcp:8080",
            "MCP_SMOKE_SOURCE_ID": "source",
            "MCP_SMOKE_TARGET_ID": "target",
        },
    )

    config = config_from_env()

    assert config == LiveSmokeConfig(
        base_url="http://mcp:8080",
        tenant_id="tenant-live",
        repo_id="week5-e2e",
        source_id="source",
        target_id="target",
    )
