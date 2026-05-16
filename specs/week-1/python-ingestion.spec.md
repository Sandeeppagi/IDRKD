# Spec: Python Ingestion

## Goal

Parse Python source files with Tree-sitter and emit deterministic typed records
that can be written to Neo4j.

## Inputs

- `tenant_id`
- `repo_id`
- repo-relative file path
- Python source text

## Output Contract

`parse_python_file` returns `ParsedFile` with:

- `tenant_id`
- `repo_id`
- normalized path
- language `python`
- SHA-256 content hash
- tuple of `CodeEntity`
- tuple of `CodeRelation`

Entity kinds emitted in Week 1:

- `file`
- `import`
- `function`
- `class`

Relation types emitted in Week 1:

- `IMPORTS`
- `DEFINES`
- `CONTAINS`

Each entity must include:

- deterministic `id`
- `tenant_id`
- `repo_id`
- `kind`
- `name`
- `qualified_name`
- source location
- `content_hash`
- `created_at`
- `updated_at`
- `lamport_clock`

## Implementation

- `src/idrkd/ingestion/python_extractor.py`
- `src/idrkd/common/models.py`
- `src/idrkd/common/fingerprints.py`
- `tests/unit/test_python_extractor.py`
- `tests/unit/test_fingerprints.py`

## Acceptance Criteria

- Uses `tree_sitter.Parser` and `tree_sitter_python` grammar.
- Does not use Python stdlib `ast`.
- Extracts File, Import, Function, and Class records.
- Extracts class methods as Function records.
- Preserves async/decorator/argument/base-class metadata.
- Uses SHA-256 content fingerprinting.
- IDs remain stable across path and line-ending normalization.

## Verification

```bash
rg -n "import ast|ast\\.parse" src/idrkd/ingestion/python_extractor.py
uv run pytest tests/unit/test_python_extractor.py tests/unit/test_fingerprints.py
```

Expected:

- `rg` returns no stdlib AST usage.
- Unit tests pass.
