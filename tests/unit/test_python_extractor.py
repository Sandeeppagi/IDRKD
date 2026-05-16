from idrkd.common.models import EntityKind, RelationType
from idrkd.ingestion.python_extractor import parse_python_file


SOURCE = """
import os
from collections import defaultdict


def top_level(name: str) -> str:
    return name.upper()


class Worker(BaseWorker):
    @classmethod
    def build(cls):
        return cls()

    async def run(self, item):
        return item
"""


def test_parse_python_file_extracts_core_entities() -> None:
    parsed = parse_python_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="./src/example.py",
        source=SOURCE,
    )

    kinds = [entity.kind for entity in parsed.entities]
    assert kinds.count(EntityKind.FILE) == 1
    assert kinds.count(EntityKind.IMPORT) == 2
    assert kinds.count(EntityKind.FUNCTION) == 3
    assert kinds.count(EntityKind.CLASS) == 1
    assert parsed.path == "src/example.py"
    assert parsed.language == "python"


def test_parse_python_file_emits_idempotent_relations() -> None:
    parsed_a = parse_python_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="src/example.py",
        source=SOURCE,
    )
    parsed_b = parse_python_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="./src\\example.py",
        source=SOURCE.replace("\n", "\r\n"),
    )

    assert [entity.id for entity in parsed_a.entities] == [entity.id for entity in parsed_b.entities]
    assert [relation.id for relation in parsed_a.relations] == [
        relation.id for relation in parsed_b.relations
    ]
    relation_types = {relation.relation_type for relation in parsed_a.relations}
    assert RelationType.IMPORTS in relation_types
    assert RelationType.DEFINES in relation_types
    assert RelationType.CONTAINS in relation_types

