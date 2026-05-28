"""Document extraction and lightweight SpanBERT-style NER facade."""

from __future__ import annotations

from dataclasses import dataclass
import re

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
    """NER facade with deterministic fallback for local development.

    The production implementation can swap this class for a real SpanBERT NER
    model without changing the downstream document ingestion contract.
    """

    def extract(self, text: str) -> tuple[NamedEntity, ...]:
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
