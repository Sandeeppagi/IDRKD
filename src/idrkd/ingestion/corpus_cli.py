"""Bootstrap the live release corpus into Neo4j and pgvector."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from idrkd.graph.writer import Neo4jCodeGraphWriter
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.pipeline import CommitIngestionPipeline
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import PostgresVectorStore


SUPPORTED_SUFFIXES = frozenset({".py", ".js", ".jsx", ".mjs", ".cjs", ".json", ".csv", ".md", ".markdown", ".txt"})
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


def _environment_value(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class CorpusRepository:
    repo_id: str
    local_path: Path
    snapshot_ref: str


def load_corpus_manifest(path: Path, *, repo_ids: set[str] | None = None) -> list[CorpusRepository]:
    repositories: list[CorpusRepository] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Corpus manifest line {line_number} must be an object")
        repo_id = str(value.get("id", "")).strip()
        local_path = str(value.get("local_path", "")).strip()
        snapshot_ref = str(value.get("snapshot_ref", "")).strip()
        if not repo_id or not local_path or not snapshot_ref:
            raise ValueError(f"Corpus manifest line {line_number} is missing id, local_path, or snapshot_ref")
        if repo_ids is None or repo_id in repo_ids:
            repositories.append(CorpusRepository(repo_id, Path(local_path), snapshot_ref))
    if repo_ids:
        missing = repo_ids - {repository.repo_id for repository in repositories}
        if missing:
            raise ValueError(f"Unknown repository IDs: {', '.join(sorted(missing))}")
    if not repositories:
        raise ValueError("No corpus repositories selected")
    return repositories


def discover_source_files(repo_root: Path, *, max_file_bytes: int) -> list[str]:
    paths: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(repo_root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        paths.append(relative.as_posix())
    return sorted(paths)


def ingest_corpus(args: argparse.Namespace) -> dict[str, Any]:
    repositories = load_corpus_manifest(
        args.manifest,
        repo_ids=set(args.repo_id) if args.repo_id else None,
    )
    embeddings = BgeM3EmbeddingAdapter.from_sentence_transformers(
        args.embedding_model,
        dimensions=args.embedding_dimensions,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        local_files_only=args.local_files_only,
    )
    vector_store = PostgresVectorStore(args.postgres_dsn)
    writer = Neo4jCodeGraphWriter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        writer.apply_schema()
        for repository in repositories:
            repo_root = (args.project_root / repository.local_path).resolve()
            if not repo_root.is_dir():
                raise FileNotFoundError(f"Repository snapshot does not exist: {repo_root}")
            paths = discover_source_files(repo_root, max_file_bytes=args.max_file_bytes)
            pipeline = CommitIngestionPipeline(
                repo_root=repo_root,
                writer=writer,
                embedding_sink=vector_store,
                embeddings=embeddings,
                embedding_model_name=args.embedding_model,
            )
            totals = {"parsed_files": 0, "entities": 0, "relations": 0, "embeddings": 0}
            errors: list[dict[str, str]] = []
            repo_started = time.perf_counter()
            print(f"[{repository.repo_id}] ingesting {len(paths)} files", flush=True)
            for index, path in enumerate(paths, start=1):
                event = CommitEvent.create(
                    tenant_id=args.tenant_id,
                    repo_id=repository.repo_id,
                    commit_sha=repository.snapshot_ref,
                    changed_paths=(path,),
                )
                try:
                    result = pipeline.process(
                        event,
                        correlation_id=f"corpus-{repository.repo_id}-{index}",
                        apply_schema=False,
                    )
                    for key in totals:
                        totals[key] += int(getattr(result, key))
                except Exception as exc:
                    errors.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
                    if args.strict:
                        raise
                if index % args.progress_every == 0 or index == len(paths):
                    print(
                        f"[{repository.repo_id}] {index}/{len(paths)} files, "
                        f"{totals['entities']} entities, {len(errors)} errors",
                        flush=True,
                    )
            results.append(
                {
                    **asdict(repository),
                    "local_path": str(repository.local_path),
                    "discovered_files": len(paths),
                    **totals,
                    "error_count": len(errors),
                    "errors": errors,
                    "duration_seconds": time.perf_counter() - repo_started,
                }
            )
    finally:
        writer.close()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "tenant_id": args.tenant_id,
        "embedding_model": args.embedding_model,
        "repository_count": len(results),
        "duration_seconds": time.perf_counter() - started,
        "repositories": results,
        "totals": {
            key: sum(int(result[key]) for result in results)
            for key in ("parsed_files", "entities", "relations", "embeddings", "error_count")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest the release corpus into Neo4j and pgvector.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("eval/corpus/repository_corpus.jsonl"))
    parser.add_argument("--repo-id", action="append", help="Repository ID to ingest; repeat to select several.")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument(
        "--postgres-dsn",
        default=_environment_value(
            "POSTGRES_DSN",
            "postgresql://idrkd:idrkd@localhost:5432/idrkd",
        ),
    )
    parser.add_argument(
        "--neo4j-uri",
        default=_environment_value("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument("--neo4j-user", default=_environment_value("NEO4J_USER", "neo4j"))
    parser.add_argument(
        "--neo4j-password",
        default=_environment_value("NEO4J_PASSWORD", "change-me"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-dimensions", type=int, default=1536)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-device", help="SentenceTransformer device, for example cuda or cpu.")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = ingest_corpus(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ingestion_summary={args.out} totals={json.dumps(result['totals'], sort_keys=True)}")


if __name__ == "__main__":
    main()
