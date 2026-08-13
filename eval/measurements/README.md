# Deterministic Harness Self-Tests

The artifacts in this directory use oracle predictors that return each case's expected tool call.
They verify deterministic ordering, serialization, registry execution, metrics, and bundle writing.

They are not live student, teacher, baseline, latency, memory, or comparative measurements and
must not be used to support Chapter 6 empirical claims. Live release evidence belongs under
`eval/releases/` and must identify its model endpoint, runtime, task split, and evidence scope.
