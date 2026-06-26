"""Security validation gates for MCP and A2A boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


DANGEROUS_CYPHER = re.compile(
    r"\b(CREATE|DELETE|DETACH|DROP|LOAD\s+CSV|MERGE|REMOVE|SET)\b",
    re.IGNORECASE,
)

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "exfiltrate",
)


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def require_allowed(self) -> None:
        if not self.allowed:
            raise PermissionError("; ".join(self.reasons))


def validate_tenant_scope(*, requested_tenant_id: str, principal_tenant_id: str) -> SecurityDecision:
    if requested_tenant_id == principal_tenant_id:
        return SecurityDecision(allowed=True)
    return SecurityDecision(allowed=False, reasons=("tenant scope mismatch",))


def validate_read_only_cypher(query: str) -> SecurityDecision:
    reasons: list[str] = []
    if DANGEROUS_CYPHER.search(query):
        reasons.append("cypher query is not read-only")
    if ";" in query.strip().rstrip(";"):
        reasons.append("multiple cypher statements are not allowed")
    return SecurityDecision(allowed=not reasons, reasons=tuple(reasons))


def detect_prompt_injection(text: str) -> SecurityDecision:
    lowered = text.lower()
    reasons = [
        f"prompt-injection marker detected: {marker}"
        for marker in PROMPT_INJECTION_MARKERS
        if marker in lowered
    ]
    return SecurityDecision(allowed=not reasons, reasons=tuple(reasons))


def validate_tool_payload(
    *,
    tenant_id: str,
    principal_tenant_id: str,
    query_text: str | None = None,
    cypher: str | None = None,
) -> SecurityDecision:
    decisions = [validate_tenant_scope(requested_tenant_id=tenant_id, principal_tenant_id=principal_tenant_id)]
    if query_text is not None:
        decisions.append(detect_prompt_injection(query_text))
    if cypher is not None:
        decisions.append(validate_read_only_cypher(cypher))
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    return SecurityDecision(allowed=not reasons, reasons=reasons)
