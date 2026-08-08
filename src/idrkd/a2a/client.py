"""Thin async client wrapper over a2a-sdk's client, with optional mTLS transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest

from idrkd.security.transport import TransportSecurityConfig, build_ssl_context


class IdrkdA2AClient:
    """Sends tool-call requests to a remote IDRKD (or peer) A2A agent."""

    def __init__(
        self,
        agent_card: AgentCard,
        *,
        streaming: bool = False,
        transport_security: TransportSecurityConfig | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._agent_card = agent_card
        self._streaming = streaming
        self._httpx_client = httpx_client or _build_httpx_client(transport_security)
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            config = ClientConfig(streaming=self._streaming, httpx_client=self._httpx_client)
            self._client = await create_client(agent=self._agent_card, client_config=config)
        return self._client

    async def send_message(self, text: str) -> AsyncIterator[Any]:
        client = await self._ensure_client()
        message = new_text_message(text, role=Role.ROLE_USER)
        request = SendMessageRequest(message=message)
        async for chunk in client.send_message(request):
            yield chunk

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _build_httpx_client(transport_security: TransportSecurityConfig | None) -> httpx.AsyncClient:
    if transport_security is None or not transport_security.require_mtls:
        return httpx.AsyncClient()
    return httpx.AsyncClient(verify=build_ssl_context(transport_security))
