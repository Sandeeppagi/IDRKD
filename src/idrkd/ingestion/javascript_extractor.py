"""JavaScript Tree-sitter extractor for Week 2 ingestion."""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_javascript

from idrkd.common.fingerprints import entity_id, normalise_repo_path, relation_id, sha256_text
from idrkd.common.models import (
    CodeEntity,
    CodeRelation,
    EntityKind,
    ParsedFile,
    RelationType,
    SourceLocation,
)


JAVASCRIPT_LANGUAGE = Language(tree_sitter_javascript.language())


class JavaScriptExtractionError(ValueError):
    """Raised when JavaScript source cannot be parsed."""


def parse_javascript_file(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    source: str,
) -> ParsedFile:
    repo_path = normalise_repo_path(path)
    content_hash = sha256_text(source)
    source_bytes = source.encode("utf-8")
    parser = Parser()
    parser.language = JAVASCRIPT_LANGUAGE
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        raise JavaScriptExtractionError(f"Cannot parse {repo_path}: Tree-sitter reported errors")

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

    for node in _walk_named(tree.root_node):
        if node.parent is not None and node.parent.type in {"program", "export_statement"}:
            if node.type == "import_statement":
                import_entity = _entity(
                    tenant_id=tenant_id,
                    repo_id=repo_id,
                    path=repo_path,
                    kind=EntityKind.IMPORT,
                    name=_import_name(node, source_bytes),
                    qualified_name=f"{repo_path}:{_import_name(node, source_bytes)}",
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    content_hash=content_hash,
                )
                entities.append(import_entity)
                relations.append(_relation(tenant_id, repo_id, file_entity.id, RelationType.IMPORTS, import_entity.id))
            elif node.type in {"function_declaration", "generator_function_declaration"}:
                function_entity = _function_entity(
                    tenant_id,
                    repo_id,
                    repo_path,
                    content_hash,
                    node,
                    source_bytes,
                    _node_name(node, source_bytes),
                )
                entities.append(function_entity)
                relations.append(_relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, function_entity.id))
            elif node.type == "class_declaration":
                class_name = _node_name(node, source_bytes)
                class_entity = _entity(
                    tenant_id=tenant_id,
                    repo_id=repo_id,
                    path=repo_path,
                    kind=EntityKind.CLASS,
                    name=class_name,
                    qualified_name=f"{_module_name(repo_path)}.{class_name}",
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    content_hash=content_hash,
                    properties={"bases": _class_bases(node, source_bytes)},
                )
                entities.append(class_entity)
                relations.append(_relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, class_entity.id))
                for method in _class_methods(node):
                    method_name = _node_name(method, source_bytes)
                    method_entity = _function_entity(
                        tenant_id,
                        repo_id,
                        repo_path,
                        content_hash,
                        method,
                        source_bytes,
                        f"{class_name}.{method_name}",
                    )
                    entities.append(method_entity)
                    relations.append(_relation(tenant_id, repo_id, class_entity.id, RelationType.CONTAINS, method_entity.id))

    return ParsedFile(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=repo_path,
        language="javascript",
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
        language="javascript",
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
    node: Node,
    source_bytes: bytes,
    qualified_suffix: str,
) -> CodeEntity:
    return _entity(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=path,
        kind=EntityKind.FUNCTION,
        name=_node_name(node, source_bytes),
        qualified_name=f"{_module_name(path)}.{qualified_suffix}",
        start_line=_start_line(node),
        end_line=_end_line(node),
        content_hash=content_hash,
        properties={"async": _is_async(node), "args": _parameter_names(node, source_bytes), "decorators": []},
    )


def _walk_named(node: Node) -> list[Node]:
    nodes = [node]
    for child in node.named_children:
        nodes.extend(_walk_named(child))
    return nodes


def _class_methods(node: Node) -> list[Node]:
    body = node.child_by_field_name("body")
    if body is None:
        return []
    return [child for child in body.named_children if child.type == "method_definition"]


def _class_bases(node: Node, source_bytes: bytes) -> list[str]:
    superclass = node.child_by_field_name("superclass")
    return [_text(superclass, source_bytes)] if superclass is not None else []


def _parameter_names(node: Node, source_bytes: bytes) -> list[str]:
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return []
    return [
        _text(child, source_bytes)
        for child in parameters.named_children
        if child.type in {"identifier", "rest_pattern"}
    ]


def _import_name(node: Node, source_bytes: bytes) -> str:
    source_node = node.child_by_field_name("source")
    if source_node is not None:
        return _text(source_node, source_bytes).strip("'\"")
    return _text(node, source_bytes)


def _node_name(node: Node, source_bytes: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, source_bytes)
    for child in node.named_children:
        if child.type in {"identifier", "property_identifier"}:
            return _text(child, source_bytes)
    return _text(node, source_bytes)


def _module_name(path: str) -> str:
    without_ext = path[:-3] if path.endswith(".js") else path
    return without_ext.replace("/", ".")


def _is_async(node: Node) -> bool:
    return any(child.type == "async" for child in node.children)


def _start_line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    return node.end_point[0] + 1


def _text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
