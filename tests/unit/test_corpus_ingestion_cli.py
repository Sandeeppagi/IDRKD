from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from idrkd.ingestion.corpus_cli import discover_source_files, load_corpus_manifest
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(self, texts, *, batch_size, show_progress_bar):
        self.calls.append((texts, batch_size, show_progress_bar))
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


def test_embedding_adapter_batches_and_pads_model_vectors() -> None:
    model = _BatchModel()
    embeddings = BgeM3EmbeddingAdapter(dimensions=4, model=model, batch_size=8)

    vectors = embeddings.embed_many(["one", "two"])

    assert model.calls == [(["one", "two"], 8, False)]
    assert vectors == [[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]


def test_manifest_can_select_repository_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "repo-a",
                "local_path": "data/a",
                "snapshot_ref": "abc",
                "source_url": "https://example.test/a.git",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "repo-b",
                "local_path": "data/b",
                "snapshot_ref": "def",
                "source_url": "https://example.test/b.git",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    repositories = load_corpus_manifest(manifest, repo_ids={"repo-b"})

    assert [repository.repo_id for repository in repositories] == ["repo-b"]
    assert repositories[0].source_url == "https://example.test/b.git"
    with pytest.raises(ValueError, match="repo-c"):
        load_corpus_manifest(manifest, repo_ids={"repo-c"})


def test_source_discovery_filters_generated_and_oversized_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Readme\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    (tmp_path / "large.json").write_text("x" * 20, encoding="utf-8")

    paths = discover_source_files(tmp_path, max_file_bytes=10)

    assert paths == ["README.md"]
