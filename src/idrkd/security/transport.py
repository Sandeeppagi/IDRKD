"""mTLS transport configuration for A2A cross-process calls."""

from __future__ import annotations

from dataclasses import dataclass
import ssl


@dataclass(frozen=True)
class TransportSecurityConfig:
    require_mtls: bool = False
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    verify_hostname: bool = True


def build_ssl_context(config: TransportSecurityConfig) -> ssl.SSLContext:
    if not config.require_mtls:
        raise ValueError("build_ssl_context requires require_mtls=True")
    if not config.ca_cert_path or not config.client_cert_path or not config.client_key_path:
        raise ValueError("mTLS requires ca_cert_path, client_cert_path, and client_key_path")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = config.verify_hostname
    context.load_verify_locations(cafile=config.ca_cert_path)
    context.load_cert_chain(certfile=config.client_cert_path, keyfile=config.client_key_path)
    return context
