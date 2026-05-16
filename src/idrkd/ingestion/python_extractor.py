"""Python source extractor for the Week 1 ingestion MVP.

The HLD calls for Tree-sitter. This module uses Python's standard AST for the
first executable slice so the pipeline can be tested without native grammar
packages. The public output records are parser-agnostic and can be fed by a
Tree-sitter implementation later.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from idrkd.common.fingerprints import entity_id, normalise_repo_path, relation_id, sha256_text
from idrkd.common.models import (
    CodeEntity,
    CodeRelation,
    EntityKind,
    ParsedFile,
    RelationType,
    SourceLocation,
)


class PythonExtractionError(ValueError):
    """Raised when Python source cannot be parsed."""


def parse_python_file(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    source: str,
) -> ParsedFile:
    """Parse Python source into deterministic entity and relation records."""
    repo_path = normalise_repo_path(path)
    content_hash = sha256_text(source)
    try:
        tree = ast.parse(source, filename=repo_path)
    except SyntaxError as exc:
        raise PythonExtractionError(f"Cannot parse {repo_path}: {exc}") from exc

    file_entity = _entity(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=repo_path,
        kind=EntityKind.FILE,
        name=repo_path.rsplit("/", 1)[-1],
        qualified_name=repo_path,
        start_line=1,
        end_line=max(1, len(source.splitlines())),
        content_hash=content_hash,
        properties={"module": _module_name(repo_path)},
    )

    entities: list[CodeEntity] = [file_entity]
    relations: list[CodeRelation] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_name = _import_name(node)
            import_entity = _entity(
                tenant_id=tenant_id,
                repo_id=repo_id,
                path=repo_path,
                kind=EntityKind.IMPORT,
                name=import_name,
                qualified_name=f"{repo_path}:{import_name}",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                content_hash=content_hash,
            )
            entities.append(import_entity)
            relations.append(
                _relation(
                    tenant_id,
                    repo_id,
                    file_entity.id,
                    RelationType.IMPORTS,
                    import_entity.id,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entities.append(
                _function_entity(tenant_id, repo_id, repo_path, content_hash, node, node.name)
            )
            relations.append(_relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, entities[-1].id))
        elif isinstance(node, ast.ClassDef):
            class_entity = _entity(
                tenant_id=tenant_id,
                repo_id=repo_id,
                path=repo_path,
                kind=EntityKind.CLASS,
                name=node.name,
                qualified_name=f"{_module_name(repo_path)}.{node.name}",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                content_hash=content_hash,
                properties={"bases": [_name(base) for base in node.bases]},
            )
            entities.append(class_entity)
            relations.append(
                _relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, class_entity.id)
            )
            for child in _class_methods(node):
                method_entity = _function_entity(
                    tenant_id,
                    repo_id,
                    repo_path,
                    content_hash,
                    child,
                    f"{node.name}.{child.name}",
                )
                entities.append(method_entity)
                relations.append(
                    _relation(
                        tenant_id,
                        repo_id,
                        class_entity.id,
                        RelationType.CONTAINS,
                        method_entity.id,
                    )
                )

    return ParsedFile(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=repo_path,
        language="python",
        content_hash=content_hash,
        entities=tuple(entities),
        relations=tuple(relations),
    )


def _entity(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    kind: EntityKind,
    name: str,
    qualified_name: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    properties: dict[str, object] | None = None,
) -> CodeEntity:
    return CodeEntity(
        id=entity_id(tenant_id, repo_id, path, kind.value, qualified_name),
        tenant_id=tenant_id,
        repo_id=repo_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        location=SourceLocation(path=path, start_line=start_line, end_line=end_line),
        content_hash=content_hash,
        language="python",
        properties=properties or {},
    )


def _relation(
    tenant_id: str,
    repo_id: str,
    source_id: str,
    relation_type: RelationType,
    target_id: str,
) -> CodeRelation:
    return CodeRelation(
        id=relation_id(tenant_id, repo_id, source_id, relation_type.value, target_id),
        tenant_id=tenant_id,
        repo_id=repo_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
    )


def _function_entity(
    tenant_id: str,
    repo_id: str,
    path: str,
    content_hash: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_suffix: str,
) -> CodeEntity:
    args = [arg.arg for arg in node.args.args]
    return _entity(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=path,
        kind=EntityKind.FUNCTION,
        name=node.name,
        qualified_name=f"{_module_name(path)}.{qualified_suffix}",
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        content_hash=content_hash,
        properties={
            "async": isinstance(node, ast.AsyncFunctionDef),
            "args": args,
            "decorators": [_name(dec) for dec in node.decorator_list],
        },
    )


def _class_methods(node: ast.ClassDef) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    return (child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _module_name(path: str) -> str:
    without_ext = path[:-3] if path.endswith(".py") else path
    return without_ext.replace("/", ".")


def _import_name(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    module = "." * node.level + (node.module or "")
    imported = ",".join(alias.name for alias in node.names)
    return f"{module}:{imported}"


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return node.__class__.__name__

