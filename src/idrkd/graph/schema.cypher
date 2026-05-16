// Week 1 Neo4j schema for IDRKD structural ingestion.

CREATE CONSTRAINT code_entity_id IF NOT EXISTS
FOR (n:CodeEntity)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT file_entity_id IF NOT EXISTS
FOR (n:File)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT module_entity_id IF NOT EXISTS
FOR (n:Module)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT class_entity_id IF NOT EXISTS
FOR (n:Class)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT function_entity_id IF NOT EXISTS
FOR (n:Function)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT import_entity_id IF NOT EXISTS
FOR (n:Import)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT schema_entity_id IF NOT EXISTS
FOR (n:Schema)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT document_entity_id IF NOT EXISTS
FOR (n:Document)
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

CREATE INDEX file_tenant_repo IF NOT EXISTS
FOR (n:File)
ON (n.tenant_id, n.repo_id);

CREATE INDEX module_tenant_repo IF NOT EXISTS
FOR (n:Module)
ON (n.tenant_id, n.repo_id);

CREATE INDEX class_tenant_repo IF NOT EXISTS
FOR (n:Class)
ON (n.tenant_id, n.repo_id);

CREATE INDEX function_tenant_repo IF NOT EXISTS
FOR (n:Function)
ON (n.tenant_id, n.repo_id);

CREATE INDEX import_tenant_repo IF NOT EXISTS
FOR (n:Import)
ON (n.tenant_id, n.repo_id);

CREATE INDEX schema_tenant_repo IF NOT EXISTS
FOR (n:Schema)
ON (n.tenant_id, n.repo_id);

CREATE INDEX document_tenant_repo IF NOT EXISTS
FOR (n:Document)
ON (n.tenant_id, n.repo_id);

CREATE INDEX code_entity_created_at IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.created_at);

CREATE INDEX code_entity_updated_at IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.updated_at);

CREATE INDEX code_entity_lamport_clock IF NOT EXISTS
FOR (n:CodeEntity)
ON (n.lamport_clock);

CREATE CONSTRAINT relation_id IF NOT EXISTS
FOR ()-[r:RELATES_TO]-()
REQUIRE r.id IS UNIQUE;
