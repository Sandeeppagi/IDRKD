"""Document extraction and optional SpanBERT-style NER."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any, cast

from idrkd.common.fingerprints import entity_id, normalise_repo_path, sha256_text
from idrkd.common.models import CodeEntity, EntityKind, ParsedFile, SourceLocation


ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*|[A-Z]{2,})(?:\s+[A-Z][a-z0-9]+)*\b")


@dataclass(frozen=True)
class NamedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float


class SpanBertNerExtractor:
    """NER extractor with optional Hugging Face token-classification pipeline.

    When `pipeline` is omitted, deterministic regex extraction keeps local tests
    and offline development repeatable.
    """

    def __init__(self, pipeline: object | None = None) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_transformers(
        cls,
        model_name: str = "SpanBERT/spanbert-large-cased",
        *,
        local_files_only: bool = False,
    ) -> SpanBertNerExtractor:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        build_pipeline = cast(Callable[..., object], pipeline)
        return cls(
            build_pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        )

    def extract(self, text: str) -> tuple[NamedEntity, ...]:
        if self._pipeline is not None:
            return tuple(_entities_from_pipeline(self._pipeline, text))
        entities = []
        for match in ENTITY_PATTERN.finditer(text):
            token = match.group(0)
            label = "ORG" if token.isupper() or " " in token else "TERM"
            entities.append(NamedEntity(token, label, match.start(), match.end(), 0.55))
        return tuple(entities)


def parse_document_file(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    source: str,
    ner: SpanBertNerExtractor | None = None,
) -> ParsedFile:
    repo_path = normalise_repo_path(path)
    content_hash = sha256_text(source)
    extractor = ner or SpanBertNerExtractor()
    named_entities = [
        {
            "text": entity.text,
            "label": entity.label,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence,
        }
        for entity in extractor.extract(source)
    ]
    entity = CodeEntity(
        id=entity_id(tenant_id, repo_id, repo_path, EntityKind.DOCUMENT.value, repo_path),
        tenant_id=tenant_id,
        repo_id=repo_id,
        kind=EntityKind.DOCUMENT,
        name=repo_path.rsplit("/", 1)[-1],
        qualified_name=repo_path,
        location=SourceLocation(path=repo_path, start_line=1, end_line=max(1, len(source.splitlines()))),
        content_hash=content_hash,
        language="markdown" if repo_path.endswith(".md") else "document",
        properties={"named_entities": named_entities},
    )
    return ParsedFile(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=repo_path,
        language=entity.language,
        content_hash=content_hash,
        entities=(entity,),
        relations=(),
    )


def _entities_from_pipeline(pipeline_model: object, text: str) -> list[NamedEntity]:
    ner = cast(Callable[[str], list[dict[str, Any]]], pipeline_model)
    raw_entities = ner(text)
    entities: list[NamedEntity] = []
    for raw in raw_entities:
        entity_text = str(raw.get("word") or raw.get("entity") or "").strip()
        if not entity_text:
            continue
        start = int(raw.get("start", 0))
        end = int(raw.get("end", start + len(entity_text)))
        label = str(raw.get("entity_group") or raw.get("entity") or "ENTITY")
        confidence = float(raw.get("score", 0.0))
        entities.append(NamedEntity(entity_text, label, start, end, confidence))
    return entities
