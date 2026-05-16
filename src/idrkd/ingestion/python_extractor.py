"""Python Tree-sitter source extractor for the Week 1 ingestion MVP."""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_python

from idrkd.common.fingerprints import entity_id, normalise_repo_path, relation_id, sha256_text
from idrkd.common.models import (
    CodeEntity,
    CodeRelation,
    EntityKind,
    ParsedFile,
    RelationType,
    SourceLocation,
)


PYTHON_LANGUAGE = Language(tree_sitter_python.language())


class PythonExtractionError(ValueError):
    """Raised when Python source cannot be parsed."""


def parse_python_file(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    source: str,
) -> ParsedFile:
    """Parse Python source into deterministic Tree-sitter-backed records."""
    repo_path = normalise_repo_path(path)
    content_hash = sha256_text(source)
    source_bytes = source.encode("utf-8")
    parser = Parser()
    parser.language = PYTHON_LANGUAGE
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        raise PythonExtractionError(f"Cannot parse {repo_path}: Tree-sitter reported errors")

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

    for node in tree.root_node.named_children:
        definition_node, decorators = _unwrap_decorated_definition(node, source_bytes)
        if definition_node.type in {"import_statement", "import_from_statement"}:
            import_name = _import_name(definition_node, source_bytes)
            import_entity = _entity(
                tenant_id=tenant_id,
                repo_id=repo_id,
                path=repo_path,
                kind=EntityKind.IMPORT,
                name=import_name,
                qualified_name=f"{repo_path}:{import_name}",
                start_line=_start_line(definition_node),
                end_line=_end_line(definition_node),
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
        elif definition_node.type == "function_definition":
            entities.append(
                _function_entity(
                    tenant_id,
                    repo_id,
                    repo_path,
                    content_hash,
                    definition_node,
                    source_bytes,
                    _node_name(definition_node, source_bytes),
                    decorators,
                )
            )
            relations.append(_relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, entities[-1].id))
        elif definition_node.type == "class_definition":
            class_name = _node_name(definition_node, source_bytes)
            class_entity = _entity(
                tenant_id=tenant_id,
                repo_id=repo_id,
                path=repo_path,
                kind=EntityKind.CLASS,
                name=class_name,
                qualified_name=f"{_module_name(repo_path)}.{class_name}",
                start_line=_start_line(definition_node),
                end_line=_end_line(definition_node),
                content_hash=content_hash,
                properties={"bases": _class_bases(definition_node, source_bytes)},
            )
            entities.append(class_entity)
            relations.append(
                _relation(tenant_id, repo_id, file_entity.id, RelationType.DEFINES, class_entity.id)
            )
            for child, child_decorators in _class_methods(definition_node, source_bytes):
                child_name = _node_name(child, source_bytes)
                method_entity = _function_entity(
                    tenant_id,
                    repo_id,
                    repo_path,
                    content_hash,
                    child,
                    source_bytes,
                    f"{class_name}.{child_name}",
                    child_decorators,
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
    node: Node,
    source_bytes: bytes,
    qualified_suffix: str,
    decorators: list[str] | None = None,
) -> CodeEntity:
    function_name = _node_name(node, source_bytes)
    return _entity(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=path,
        kind=EntityKind.FUNCTION,
        name=function_name,
        qualified_name=f"{_module_name(path)}.{qualified_suffix}",
        start_line=_start_line(node),
        end_line=_end_line(node),
        content_hash=content_hash,
        properties={
            "async": _is_async_function(node),
            "args": _parameter_names(node, source_bytes),
            "decorators": decorators or [],
        },
    )


def _module_name(path: str) -> str:
    without_ext = path[:-3] if path.endswith(".py") else path
    return without_ext.replace("/", ".")


def _class_methods(node: Node, source_bytes: bytes) -> list[tuple[Node, list[str]]]:
    body = node.child_by_field_name("body")
    if body is None:
        return []

    methods: list[tuple[Node, list[str]]] = []
    for child in body.named_children:
        definition_node, decorators = _unwrap_decorated_definition(child, source_bytes)
        if definition_node.type == "function_definition":
            methods.append((definition_node, decorators))
    return methods


def _unwrap_decorated_definition(node: Node, source_bytes: bytes) -> tuple[Node, list[str]]:
    if node.type != "decorated_definition":
        return node, []

    decorators = [
        _decorator_name(child, source_bytes)
        for child in node.named_children
        if child.type == "decorator"
    ]
    definition = node.child_by_field_name("definition")
    return definition or node, decorators


def _import_name(node: Node, source_bytes: bytes) -> str:
    if node.type == "import_statement":
        return ",".join(
            _text(child, source_bytes)
            for child in node.named_children
            if child.type in {"dotted_name", "aliased_import"}
        )

    module_parts: list[str] = []
    imported_parts: list[str] = []
    seen_import_keyword = False
    for child in node.children:
        if child.type == "import":
            seen_import_keyword = True
            continue
        if not child.is_named:
            if child.type == "." and not seen_import_keyword:
                module_parts.append(".")
            continue
        if child.type in {"dotted_name", "aliased_import", "wildcard_import"}:
            target = _text(child, source_bytes)
            if seen_import_keyword:
                imported_parts.append(target)
            else:
                module_parts.append(target)

    return f"{''.join(module_parts)}:{','.join(imported_parts)}"


def _node_name(node: Node, source_bytes: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, source_bytes)
    for child in node.named_children:
        if child.type == "identifier":
            return _text(child, source_bytes)
    return _text(node, source_bytes)


def _class_bases(node: Node, source_bytes: bytes) -> list[str]:
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is None:
        return []
    return [
        _text(child, source_bytes)
        for child in superclasses.named_children
        if child.type not in {"keyword_argument", "dictionary_splat", "list_splat"}
    ]


def _parameter_names(node: Node, source_bytes: bytes) -> list[str]:
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return []

    names: list[str] = []
    for child in parameters.named_children:
        name = _first_identifier(child)
        if name is not None:
            names.append(_text(name, source_bytes))
    return names


def _first_identifier(node: Node) -> Node | None:
    if node.type == "identifier":
        return node
    for child in node.named_children:
        identifier = _first_identifier(child)
        if identifier is not None:
            return identifier
    return None


def _decorator_name(node: Node, source_bytes: bytes) -> str:
    named_children = list(node.named_children)
    if not named_children:
        return _text(node, source_bytes).lstrip("@")
    return _text(named_children[0], source_bytes)


def _is_async_function(node: Node) -> bool:
    return any(child.type == "async" for child in node.children)


def _start_line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    return node.end_point[0] + 1


def _text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
