"""vLLM serving contract for the distilled student model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from collections.abc import Mapping
from typing import Protocol
from urllib import request


@dataclass(frozen=True)
class VllmServingConfig:
    model_path: str
    served_model_name: str = "idrkd-phi4-mini"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9

    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> tuple[str, ...]:
        return (
            "vllm",
            "serve",
            self.model_path,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        )


@dataclass(frozen=True)
class OllamaServingConfig:
    model_name: str = "idrkd-student"
    host: str = "0.0.0.0"
    port: int = 11434

    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class StudentModelClient(Protocol):
    def generate(self, *, query: str, evidence: list[str]) -> str:
        ...


@dataclass(frozen=True)
class OpenAICompatibleStudentClient:
    base_url: str
    model: str
    api_key: str = "idrkd-local"
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 256

    def generate(self, *, query: str, evidence: list[str]) -> str:
        prompt = (
            "Answer the IDRKD query using only the provided repository evidence.\n\n"
            f"Query: {query}\n"
            "Evidence:\n"
            + "\n".join(f"- {item}" for item in evidence)
        )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are the IDRKD student model. Stay grounded in the supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        choices = raw.get("choices", [])
        if not choices:
            raise RuntimeError("Student model response did not include choices")
        message = choices[0].get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Student model response was empty")
        return content


def student_model_client_from_env(
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleStudentClient | None:
    values = environ or os.environ
    base_url = values.get("IDRKD_STUDENT_MODEL_BASE_URL")
    model = values.get("IDRKD_STUDENT_MODEL_ID")
    if not base_url or not model:
        return None
    return OpenAICompatibleStudentClient(
        base_url=base_url,
        model=model,
        api_key=values.get("IDRKD_STUDENT_MODEL_API_KEY", "idrkd-local"),
        timeout_seconds=float(values.get("IDRKD_STUDENT_MODEL_TIMEOUT_SECONDS", "30")),
        max_tokens=int(values.get("IDRKD_STUDENT_MODEL_MAX_TOKENS", "256")),
    )
