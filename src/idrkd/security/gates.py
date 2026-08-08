"""Security validation gates for MCP and A2A boundaries.

Implements the Week 5 5-layer prompt-injection containment:

1. Marker denylist on inbound query text (`detect_prompt_injection`).
2. Role-impersonation detection on inbound query text (`detect_role_impersonation`).
3. Read-only Cypher enforcement (`validate_read_only_cypher`).
4. Cypher-escape hardening: reject inline string literals in generated
   Cypher, since every dynamic value in this codebase's query templates
   must be a bound `$param` (`assert_no_inline_literals`).
5. Output-side quarantine and secret-leakage scanning applied to tool
   results before they leave the registry (`screen_tool_response`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


DANGEROUS_CYPHER = re.compile(
    r"\b(CREATE|DELETE|DETACH|DROP|LOAD\s+CSV|MERGE|REMOVE|SET)\b",
    re.IGNORECASE,
)

INLINE_STRING_LITERAL = re.compile(r"""['"]""")

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "exfiltrate",
)

ROLE_IMPERSONATION_MARKERS = (
    "system:",
    "developer:",
    "[system]",
    "[developer]",
    "assistant:",
)

UNTRUSTED_DATA_DELIMITER = "<<<IDRKD_UNTRUSTED_DATA>>>"


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


def detect_role_impersonation(text: str) -> SecurityDecision:
    lowered = text.lower()
    reasons = [
        f"role-impersonation marker detected: {marker}"
        for marker in ROLE_IMPERSONATION_MARKERS
        if marker in lowered
    ]
    return SecurityDecision(allowed=not reasons, reasons=tuple(reasons))


def assert_no_inline_literals(cypher: str) -> SecurityDecision:
    if INLINE_STRING_LITERAL.search(cypher):
        return SecurityDecision(
            allowed=False,
            reasons=("cypher query contains an inline string literal instead of a bound $param",),
        )
    return SecurityDecision(allowed=True)


def quarantine_untrusted_text(text: str) -> str:
    return f"{UNTRUSTED_DATA_DELIMITER}\n{text}\n{UNTRUSTED_DATA_DELIMITER}"


def detect_quarantine_breakout(text: str) -> SecurityDecision:
    if UNTRUSTED_DATA_DELIMITER in text:
        return SecurityDecision(
            allowed=False,
            reasons=("untrusted text attempts to break out of the quarantine fence",),
        )
    return SecurityDecision(allowed=True)


def scan_for_secret_leakage(text: str, known_secrets: tuple[str, ...]) -> SecurityDecision:
    reasons = tuple(
        f"response contains a known secret value ({secret[:4]}...)"
        for secret in known_secrets
        if secret and secret in text
    )
    return SecurityDecision(allowed=not reasons, reasons=reasons)


def screen_tool_response(text: str, *, known_secrets: tuple[str, ...] = ()) -> tuple[str, SecurityDecision]:
    """Layer 5: quarantine-fence and secret-scan a tool result before it leaves the registry."""
    decisions = [detect_quarantine_breakout(text), scan_for_secret_leakage(text, known_secrets)]
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    return quarantine_untrusted_text(text), SecurityDecision(allowed=not reasons, reasons=reasons)


def validate_tool_argument_allowlist(allowed_keys: frozenset[str], params: dict[str, object]) -> SecurityDecision:
    extra = set(params) - allowed_keys
    if extra:
        return SecurityDecision(allowed=False, reasons=(f"unexpected argument(s): {', '.join(sorted(extra))}",))
    return SecurityDecision(allowed=True)


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
        decisions.append(detect_role_impersonation(query_text))
    if cypher is not None:
        decisions.append(validate_read_only_cypher(cypher))
        decisions.append(assert_no_inline_literals(cypher))
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    return SecurityDecision(allowed=not reasons, reasons=reasons)
