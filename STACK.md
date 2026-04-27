# AI Safety Research Stack

`safety-probe` is layer 1 of a planned three-repo stack:

```text
safety-probe -> adaptive-redteam -> realtime-safety-monitor
```

## safety-probe

Reusable evaluation toolkit.

Provides:
- Probe schemas and probe libraries
- Safety categories and failure taxonomy
- Backend abstractions for hosted, local, and vLLM inference
- Rule, classifier, LLM, and composite judges
- Sweep orchestration for inference-time parameter studies
- Metrics, plots, reports, and mitigation profiles

## adaptive-redteam

Research system that depends on this toolkit.

Consumes:
- `FailureMode`
- `BaseBackend` / `GenerationConfig`
- Judge concepts and evaluation result structure

Produces:
- High-risk prompts
- Scored prompt-response records
- Mutation and iteration metrics
- Reports for human review

## realtime-safety-monitor

Runtime monitoring interface system. A minimal skeleton now exists at
`../realtime-safety-monitor`.

Consumes:
- Failure categories from this repo
- Scorer and judge contracts from this repo
- High-risk patterns discovered by `adaptive-redteam` (planned integration)

This stack is experimental research infrastructure. Results characterize observed behavior under specific probes, models, judges, and inference settings; they do not prove general model safety.
