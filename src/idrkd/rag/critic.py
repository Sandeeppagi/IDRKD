"""DeBERTa-v3-large NLI faithfulness critic for Week 4 synthesis gating."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    entailed: bool
    claim_scores: tuple[float, ...] = ()


class FaithfulnessCritic:
    """Faithfulness critic with optional NLI model inference and lexical fallback.

    A supplied Hugging Face zero-shot/NLI pipeline is used as the primary path.
    Without it, the class approximates an AlignScore-style faithfulness score by
    measuring answer-term support in the retrieved evidence text.
    """

    def __init__(self, threshold: float = 0.78, nli_pipeline: object | None = None) -> None:
        self.threshold = threshold
        self._nli_pipeline = nli_pipeline

    @classmethod
    def from_transformers(
        cls,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        *,
        threshold: float = 0.78,
        local_files_only: bool = False,
    ) -> FaithfulnessCritic:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        build_pipeline = cast(Callable[..., object], pipeline)
        return cls(
            threshold=threshold,
            nli_pipeline=build_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
            ),
        )

    def evaluate(self, answer: str, evidence_texts: list[str]) -> FaithfulnessResult:
        claims = _atomic_claims(answer)
        if not claims:
            return FaithfulnessResult(score=0.0, entailed=False)

        if self._nli_pipeline is not None:
            evidence = " ".join(evidence_texts)
            nli = cast(Callable[..., Any], self._nli_pipeline)
            scores = tuple(
                _entailment_score(nli({"text": evidence, "text_pair": claim}, truncation=True))
                for claim in claims
            )
            score = min(scores, default=0.0)
            return FaithfulnessResult(
                score=score,
                entailed=bool(scores) and all(value >= self.threshold for value in scores),
                claim_scores=scores,
            )

        evidence_terms: set[str] = set()
        for text in evidence_texts:
            evidence_terms.update(term.lower() for term in text.replace(".", " ").split() if term.strip())
        scores = tuple(_lexical_support_score(claim, evidence_terms) for claim in claims)
        score = min(scores, default=0.0)
        return FaithfulnessResult(
            score=score,
            entailed=bool(scores) and all(value >= self.threshold for value in scores),
            claim_scores=scores,
        )


def _atomic_claims(answer: str) -> tuple[str, ...]:
    normalized = answer.replace("\n", " ").replace(";", ".")
    claims = tuple(claim.strip() for claim in normalized.split(".") if claim.strip())
    return claims or (answer.strip(),) if answer.strip() else ()


def _lexical_support_score(claim: str, evidence_terms: set[str]) -> float:
    claim_terms = {term.lower().strip(",():`\"'") for term in claim.split() if term.strip()}
    claim_terms.discard("")
    if not claim_terms:
        return 0.0
    return len(claim_terms & evidence_terms) / len(claim_terms)


def _entailment_score(result: Any) -> float:
    if isinstance(result, dict):
        labels = [str(label).lower() for label in result.get("labels", [])]
        scores = [float(score) for score in result.get("scores", [])]
        score_by_label = dict(zip(labels, scores, strict=False))
        return score_by_label.get("entailed", score_by_label.get("entailment", 0.0))

    if isinstance(result, list) and result and isinstance(result[0], list):
        return _entailment_score(result[0])

    if isinstance(result, list):
        label_scores = {
            str(item.get("label", "")).lower(): float(item.get("score", 0.0))
            for item in result
            if isinstance(item, dict)
        }
        for label, score in label_scores.items():
            if "entail" in label:
                return score
        return label_scores.get("label_2", 0.0)

    return 0.0
