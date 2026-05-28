# Spec: Schema and Document Extraction

## Goal

Extract structured metadata from JSON/CSV schema files and Markdown/doc-like
files that Graphify does not cover.

## Contract

JSON/CSV outputs:

- `EntityKind.SCHEMA`
- source format in `properties.format`
- discovered fields in `properties.fields`

Markdown/text outputs:

- `EntityKind.DOCUMENT`
- extracted named entities in `properties.named_entities`

NER contract:

- `SpanBertNerExtractor.extract(text) -> tuple[NamedEntity, ...]`
- deterministic local fallback is used for tests
- production SpanBERT can replace the fallback without changing downstream
  parser or graph contracts

## Implementation

- `src/idrkd/ingestion/schema_extractor.py`
- `src/idrkd/ingestion/document_extractor.py`
- `tests/unit/test_week2_ingestion.py`

## Acceptance Criteria

- JSON nested keys are flattened deterministically.
- CSV headers become schema fields.
- Document ingestion emits a Document entity.
- NER output includes text, label, offsets, and confidence.

## Verification

```bash
uv run pytest tests/unit/test_week2_ingestion.py
```
