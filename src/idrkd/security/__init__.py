"""Security gates for tenancy, prompt injection, safe Cypher, and transport."""

from idrkd.security.gates import (
    SecurityDecision,
    assert_no_inline_literals,
    detect_prompt_injection,
    detect_quarantine_breakout,
    detect_role_impersonation,
    quarantine_untrusted_text,
    scan_for_secret_leakage,
    screen_tool_response,
    validate_read_only_cypher,
    validate_tenant_scope,
    validate_tool_argument_allowlist,
    validate_tool_payload,
)
from idrkd.security.transport import TransportSecurityConfig, build_ssl_context

__all__ = [
    "SecurityDecision",
    "TransportSecurityConfig",
    "assert_no_inline_literals",
    "build_ssl_context",
    "detect_prompt_injection",
    "detect_quarantine_breakout",
    "detect_role_impersonation",
    "quarantine_untrusted_text",
    "scan_for_secret_leakage",
    "screen_tool_response",
    "validate_read_only_cypher",
    "validate_tenant_scope",
    "validate_tool_argument_allowlist",
    "validate_tool_payload",
]
