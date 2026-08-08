"""Evaluation harnesses for MCP-TaskBench and BFCL-style scoring."""

from idrkd.evaluation.bfcl import FunctionCallPrediction, ToolCallMetrics, score_function_calls
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
    "McpTask",
    "BenchmarkMode",
    "ModelAgentPredictionResult",
    "OpenAICompatibleToolCallPredictor",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionInputs",
    "TaskBenchRunner",
    "ToolCallPredictor",
    "ToolCallMetrics",
    "evaluate_promotion",
    "load_tasks_jsonl",
    "parse_tool_call",
    "score_function_calls",
]
