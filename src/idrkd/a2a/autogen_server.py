"""CLI service exposing the AutoGen reconciler over A2A JSON-RPC."""

from __future__ import annotations

import argparse
import os

import uvicorn

from idrkd.a2a.agent_card import build_idrkd_agent_card
from idrkd.a2a.autogen_reconciler import (
    AutoGenReconciliationAgent,
    AutoGenReconciliationExecutor,
)
from idrkd.a2a.server import build_a2a_app
from idrkd.mcp.server import build_registry_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the IDRKD AutoGen reconciler over A2A.")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--public-endpoint",
        default=os.getenv("IDRKD_AUTOGEN_A2A_ENDPOINT", "http://127.0.0.1:8090/"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = build_registry_from_env()
    card = build_idrkd_agent_card(
        name="IDRKD AutoGen Reconciler",
        description="Runs tenant-scoped conflict reconciliation through MCP.",
        version="0.1.0",
        endpoint=args.public_endpoint,
        capabilities=("reconcile",),
    )
    app = build_a2a_app(
        card,
        agent_executor=AutoGenReconciliationExecutor(AutoGenReconciliationAgent(registry)),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
