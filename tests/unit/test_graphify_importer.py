from idrkd.graph.graphify_importer import build_import_batch


def test_build_import_batch_preserves_edges_with_external_stubs() -> None:
    graph = {
        "nodes": [
            {
                "id": "file_a",
                "label": "a.py",
                "file_type": "code",
                "source_file": "a.py",
                "source_location": "L1",
            }
        ],
        "links": [
            {
                "source": "file_a",
                "target": "external_module",
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": "a.py",
                "source_location": "L2",
                "weight": 1.0,
            }
        ],
    }

    batch = build_import_batch(
        graph,
        tenant_id="tenant-a",
        repo_id="repo-a",
        source_label="test",
    )

    assert batch.graphify_node_count == 1
    assert batch.unique_graphify_node_count == 1
    assert batch.graphify_edge_count == 1
    assert batch.external_stub_count == 1
    assert len(batch.nodes) == 2
    assert len(batch.relationships) == 1
    assert {node["file_type"] for node in batch.nodes} == {"code", "external"}
    assert batch.relationships[0]["source_id"].endswith(":file_a")
    assert batch.relationships[0]["target_id"].endswith(":external_module")
