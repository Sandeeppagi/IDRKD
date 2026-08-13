from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import pytest

from idrkd.ingestion.document_extractor import SpanBertNerExtractor
from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.retrieval import HybridHit


pytestmark = pytest.mark.skipif(
    os.getenv("IDRKD_RUN_ML_TESTS") != "1",
    reason="set IDRKD_RUN_ML_TESTS=1 to run cache-only ML integration tests",
)

T = TypeVar("T")


def _cached_or_skip(factory: Callable[[], T], model_name: str) -> T:
    try:
        return factory()
    except (OSError, ValueError) as exc:
        pytest.skip(f"{model_name!r} is not available in the local model cache: {exc}")


def test_bge_m3_embedding_adapter_runs_cached_real_model() -> None:
    pytest.importorskip("sentence_transformers")
    model_name = os.getenv("IDRKD_BGE_M3_MODEL", "BAAI/bge-m3")
    adapter = _cached_or_skip(
        lambda: BgeM3EmbeddingAdapter.from_sentence_transformers(
            model_name,
            dimensions=1024,
            local_files_only=True,
        ),
        model_name,
    )

    vector = adapter.embed("Customer API reconciles billing records.")

    assert len(vector) == 1024
    assert any(value != 0.0 for value in vector)


def test_minilm_reranker_runs_cached_real_cross_encoder() -> None:
    pytest.importorskip("sentence_transformers")
    model_name = os.getenv("IDRKD_MINILM_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = _cached_or_skip(
        lambda: MiniLmReranker.from_sentence_transformers(
            model_name,
            local_files_only=True,
        ),
        model_name,
    )
    hits = [HybridHit("customer-api", 0.0), HybridHit("billing-worker", 0.0)]
    labels = {
        "customer-api": "Customer API reconciles billing records",
        "billing-worker": "Kafka task that archives stale files",
    }

    reranked = reranker.rerank("Which component reconciles billing records?", hits, labels)

    assert [hit.entity_id for hit in reranked] == ["customer-api", "billing-worker"]


def test_deberta_critic_runs_cached_real_nli_model() -> None:
    pytest.importorskip("transformers")
    model_name = os.getenv("IDRKD_DEBERTA_CRITIC_MODEL", "cross-encoder/nli-deberta-v3-large")
    critic = _cached_or_skip(
        lambda: FaithfulnessCritic.from_transformers(
            model_name,
            threshold=0.5,
            local_files_only=True,
        ),
        model_name,
    )

    result = critic.evaluate(
        "The Customer API reconciles billing records.",
        ["The Customer API reconciles billing records for enterprise tenants."],
    )

    assert 0.0 <= result.score <= 1.0
    assert result.entailed is True


def test_ner_pipeline_runs_cached_real_token_classifier() -> None:
    pytest.importorskip("transformers")
    model_name = os.getenv("IDRKD_NER_MODEL", "dslim/bert-base-NER")
    extractor = _cached_or_skip(
        lambda: SpanBertNerExtractor.from_transformers(
            model_name,
            local_files_only=True,
        ),
        model_name,
    )

    entities = extractor.extract("Telstra uses Neo4j in Melbourne.")

    assert entities
    assert all(entity.text for entity in entities)
