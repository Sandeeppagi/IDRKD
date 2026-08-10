"""Evaluation harnesses for MCP-TaskBench and BFCL-style scoring."""

from idrkd.evaluation.bfcl import FunctionCallPrediction, ToolCallMetrics, score_function_calls
from idrkd.evaluation.measurements import MeasurementJob, OracleToolCallPredictor, build_measurement_bundle
from idrkd.evaluation.model_agent import (
    ModelAgentPredictionResult,
    OpenAICompatibleToolCallPredictor,
    ToolCallPredictor,
    parse_tool_call,
)
from idrkd.evaluation.promotion import (
    PromotionCriteria,
    PromotionDecision,
    PromotionInputs,
    evaluate_promotion,
)
from idrkd.evaluation.synthetic_schemas import (
    SyntheticSchemaCorpus,
    SyntheticSchemaGraphBackend,
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
)
from idrkd.evaluation.taskbench import (
    BenchmarkMode,
    EvalCaseResult,
    EvalSummary,
    McpTask,
    TaskBenchRunner,
    load_tasks_jsonl,
)

__all__ = [
    "EvalCaseResult",
    "EvalSummary",
    "FunctionCallPrediction",
    "MeasurementJob",
    "McpTask",
    "BenchmarkMode",
    "ModelAgentPredictionResult",
    "OpenAICompatibleToolCallPredictor",
    "OracleToolCallPredictor",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionInputs",
    "SyntheticSchemaCorpus",
    "SyntheticSchemaGraphBackend",
    "TaskBenchRunner",
    "ToolCallPredictor",
    "ToolCallMetrics",
    "build_synthetic_schema_registry",
    "build_synthetic_schema_tasks",
    "build_measurement_bundle",
    "evaluate_promotion",
    "load_synthetic_schema_corpus",
    "load_tasks_jsonl",
    "parse_tool_call",
    "score_function_calls",
]
