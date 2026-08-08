"""Builds real a2a-sdk v1.0 AgentCards for IDRKD agents."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill


def build_idrkd_agent_card(
    *,
    name: str,
    description: str,
    version: str,
    endpoint: str,
    capabilities: tuple[str, ...],
    legacy_endpoint: str | None = None,
) -> AgentCard:
    """Builds an AgentCard advertising both the v1.0 interface and, when
    `legacy_endpoint` is given, a v0.3-compatible interface — the
    backward-compat layer for legacy 0.3 servers."""
    supported_interfaces = [
        AgentInterface(protocol_binding="JSONRPC", url=endpoint, protocol_version="1.0"),
    ]
    if legacy_endpoint is not None:
        supported_interfaces.append(
            AgentInterface(protocol_binding="JSONRPC", url=legacy_endpoint, protocol_version="0.3")
        )
    skills = [
        AgentSkill(
            id=capability,
            name=capability,
            description=f"IDRKD {capability} capability",
            input_modes=["application/json"],
            output_modes=["application/json"],
        )
        for capability in capabilities
    ]
    return AgentCard(
        name=name,
        description=description,
        version=version,
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=False),
        supported_interfaces=supported_interfaces,
        skills=skills,
    )
