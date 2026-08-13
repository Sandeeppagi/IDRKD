import json
from typing import Any

import pytest

from idrkd.evaluation import model_agent
from idrkd.evaluation.model_agent import OpenAICompatibleToolCallPredictor


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_model_preflight_accepts_advertised_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_agent.request,
        "urlopen",
        lambda *args, **kwargs: _Response({"data": [{"id": "idrkd-awq"}]}),
    )
    predictor = OpenAICompatibleToolCallPredictor(
        base_url="http://127.0.0.1:8000/v1",
        model="idrkd-awq",
    )

    predictor.verify_model_available()


def test_model_preflight_rejects_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_agent.request,
        "urlopen",
        lambda *args, **kwargs: _Response({"data": [{"id": "another-model"}]}),
    )
    predictor = OpenAICompatibleToolCallPredictor(
        base_url="http://127.0.0.1:8000/v1",
        model="idrkd-awq",
    )

    with pytest.raises(RuntimeError, match="does not advertise"):
        predictor.verify_model_available()


def test_model_preflight_rejects_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(model_agent.request, "urlopen", unavailable)
    predictor = OpenAICompatibleToolCallPredictor(
        base_url="http://127.0.0.1:8000/v1",
        model="idrkd-awq",
    )

    with pytest.raises(RuntimeError, match="Cannot reach the model server"):
        predictor.verify_model_available()
