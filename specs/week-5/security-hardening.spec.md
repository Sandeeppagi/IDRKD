# Spec: Security Hardening

## Goal

Expand the single-layer prompt-injection marker denylist into the Week 5
5-layer containment chain, add Cypher-escape hardening, and add an mTLS
transport config contract for A2A cross-process calls. See
`docs/design/threat-model.md` for the STRIDE mapping this hardening
closes.

## Contract

Five layers (`src/idrkd/security/gates.py`), composed by
`validate_tool_payload` (inbound) and `screen_tool_response` (outbound):

1. **Marker denylist** — `detect_prompt_injection` (unchanged from the
   June 26 milestone).
2. **Role-impersonation detection** — `detect_role_impersonation` rejects
   text containing `"system:"`/`"developer:"`/`"[system]"`-style prefixes.
3. **Read-only Cypher enforcement** — `validate_read_only_cypher`
   (unchanged).
4. **Cypher-escape hardening** — `assert_no_inline_literals` rejects any
   generated Cypher containing a quoted string literal; every Week 5
   traversal template only ever binds dynamic values through `$param`.
5. **Output-side quarantine and secret-leakage scan** — `screen_tool_response`
   wraps a tool result in an explicit untrusted-data fence
   (`quarantine_untrusted_text`) and rejects it if it either already tries
   to break out of that fence (`detect_quarantine_breakout`) or contains a
   known secret value (`scan_for_secret_leakage`); wired into
   `McpToolRegistry.call_tool` so it runs on every tool call, not just
   ones under direct test.

`validate_tool_argument_allowlist(allowed_keys, params)` is a standalone,
reusable primitive for the same allowlisting `extra="forbid"` gives for
free on the Pydantic models in `mcp-pydantic-schemas.spec.md`.

Transport (`src/idrkd/security/transport.py`):

- `TransportSecurityConfig(require_mtls, ca_cert_path, client_cert_path, client_key_path, verify_hostname)`
- `build_ssl_context(config) -> ssl.SSLContext`: `PROTOCOL_TLS_CLIENT`,
  `verify_mode = ssl.CERT_REQUIRED`, loads the CA and client cert/key —
  wired into `IdrkdA2AClient`'s `httpx.AsyncClient(verify=...)` when
  `require_mtls=True`.

## Implementation

- `src/idrkd/security/gates.py`
- `src/idrkd/security/transport.py`
- `src/idrkd/mcp/tools.py` (`screen_tool_response` wired into `call_tool`)
- `tests/unit/test_week5_security_hardening.py`

## Acceptance Criteria

- Each of the 5 layers has an independent test showing it allows a benign
  case and rejects the corresponding adversarial case.
- `assert_no_inline_literals` allows a fully-parameterized query and
  rejects a query containing any quoted string literal.
- `build_ssl_context` raises `ValueError` when `require_mtls=False`, and
  returns a context with `verify_mode == ssl.CERT_REQUIRED` when a valid
  cert/key pair is supplied (verified against a throwaway self-signed
  certificate generated with `cryptography` in the test, not a live TLS
  handshake).

## Verification

```bash
uv run pytest tests/unit/test_week5_security_hardening.py
```
