"""DPO preference-pair construction for SLM alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from idrkd.distillation.traces import TeacherTrace


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    trace_id: str
    metadata: dict[str, Any]

    def to_dpo_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


def build_preference_pair(
    *,
    trace: TeacherTrace,
    rejected_answer: str,
    rejected_source: str = "sft_naive",
) -> PreferencePair:
    """Use the teacher trace as `chosen` and a weaker student answer as `rejected`."""

    return PreferencePair(
        prompt=trace.prompt,
        chosen=trace.answer,
        rejected=rejected_answer,
        trace_id=trace.id,
        metadata={
            "trace_id": trace.id,
            "tenant_id": trace.tenant_id,
            "repo_id": trace.repo_id,
            "rejected_source": rejected_source,
            "faithfulness_score": trace.faithfulness_score,
        },
    )
