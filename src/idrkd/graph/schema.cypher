// Week 1 Neo4j schema for IDRKD structural ingestion.

CREATE CONSTRAINT code_entity_id IF NOT EXISTS
FOR (n:CodeEntity)
REQUIRE n.id IS UNIQUE;

CREATE INDEX code_entity_tenant_repo IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.tenant_id, n.repo_id);

CREATE INDEX code_entity_kind IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.kind);

CREATE INDEX code_entity_qualified_name IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.qualified_name);

CREATE INDEX code_entity_content_hash IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.content_hash);

CREATE CONSTRAINT relation_id IF NOT EXISTS
FOR ()-[r:RELATES_TO]-()
REQUIRE r.id IS UNIQUE;

