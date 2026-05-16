from idrkd.common.fingerprints import entity_id, normalise_repo_path, relation_id, sha256_text


def test_text_hash_normalises_line_endings() -> None:
    assert sha256_text("a\r\nb\r\n") == sha256_text("a\nb\n")


def test_entity_id_is_stable_and_path_normalised() -> None:
    a = entity_id("tenant", "repo", "./pkg\\mod.py", "function", "pkg.mod.run")
    b = entity_id("tenant", "repo", "pkg/mod.py", "function", "pkg.mod.run")
    assert a == b
    assert a.startswith("ent_")


def test_relation_id_changes_with_relation_type() -> None:
    imports = relation_id("tenant", "repo", "a", "IMPORTS", "b")
    defines = relation_id("tenant", "repo", "a", "DEFINES", "b")
    assert imports != defines
    assert imports.startswith("rel_")


def test_normalise_repo_path_removes_leading_dot() -> None:
    assert normalise_repo_path("./src\\idrkd/app.py") == "src/idrkd/app.py"

