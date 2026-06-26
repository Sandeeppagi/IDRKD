"""Deterministic A2A bridge primitives with signed agent cards."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
from typing import Any
from uuid import uuid4


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class AgentCard:
    agent_id: str
    name: str
    endpoint: str
    capabilities: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class SignedAgentCard:
    card: AgentCard
    signature: str

    def verify(self, secret: str) -> bool:
        expected = sign_payload(self.card.to_payload(), secret)
        return hmac.compare_digest(expected, self.signature)


@dataclass(frozen=True)
class A2AMessage:
    sender: str
    recipient: str
    task: str
    payload: dict[str, Any]
    traceparent: str
    message_id: str = field(default_factory=lambda: f"msg_{uuid4().hex}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "task": self.task,
            "payload": self.payload,
            "traceparent": self.traceparent,
        }


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), _canonical_json(payload), hashlib.sha256).hexdigest()


def sign_agent_card(card: AgentCard, secret: str) -> SignedAgentCard:
    return SignedAgentCard(card=card, signature=sign_payload(card.to_payload(), secret))


class A2ABridge:
    def __init__(self, *, local_card: AgentCard, shared_secret: str) -> None:
        self.local_card = local_card
        self._shared_secret = shared_secret

    def signed_card(self) -> SignedAgentCard:
        return sign_agent_card(self.local_card, self._shared_secret)

    def build_message(
        self,
        *,
        recipient: str,
        task: str,
        payload: dict[str, Any],
        traceparent: str,
    ) -> A2AMessage:
        return A2AMessage(
            sender=self.local_card.agent_id,
            recipient=recipient,
            task=task,
            payload=payload,
            traceparent=traceparent,
        )

    def verify_remote_card(self, signed_card: SignedAgentCard) -> bool:
        return signed_card.verify(self._shared_secret)
