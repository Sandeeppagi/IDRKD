"""Agent-to-Agent bridge primitives on the real a2a-sdk v1.0."""

from idrkd.a2a.agent_card import build_idrkd_agent_card
from idrkd.a2a.bridge import A2ABridge, A2AMessage, SignedAgentCard, agent_card_payload, sign_agent_card
from idrkd.a2a.client import IdrkdA2AClient
from idrkd.a2a.executor import IdrkdAgentExecutor
from idrkd.a2a.server import build_a2a_app
from idrkd.a2a.task_state import A2ATaskStateStore, IdrkdTaskRecord, IdrkdTaskState

__all__ = [
    "A2ABridge",
    "A2AMessage",
    "A2ATaskStateStore",
    "IdrkdA2AClient",
    "IdrkdAgentExecutor",
    "IdrkdTaskRecord",
    "IdrkdTaskState",
    "SignedAgentCard",
    "agent_card_payload",
    "build_a2a_app",
    "build_idrkd_agent_card",
    "sign_agent_card",
]
