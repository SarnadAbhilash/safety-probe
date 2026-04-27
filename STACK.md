# Cross-Repo Interface Contracts

This document describes the interfaces that `adaptive-redteam` and `realtime-safety-monitor` consume from `safety-probe`. Changes to these interfaces require coordinated updates across all three repositories.

---

## Behavioral Failure Taxonomy

**Location:** `safety_probe.categories.failures.FailureMode`

```python
class FailureMode(str, Enum):
    SYCOPHANCY = "sycophancy"
    HALLUCINATION_UNDER_PRESSURE = "hallucination_under_pressure"
    INSTRUCTION_HIERARCHY_FAILURE = "instruction_hierarchy_failure"
    OVERCONFIDENCE_WHEN_WRONG = "overconfidence_when_wrong"
    MULTI_TURN_CONSISTENCY_FAILURE = "multi_turn_consistency_failure"
```

`adaptive-redteam` re-exports this enum as `adaptive_redteam.FailureMode`. `realtime-safety-monitor` uses the string values as category keys. Adding new values here requires corresponding category implementations in both downstream repos.

---

## Judge Interface

**Location:** `safety_probe.judges.base`

```python
class BaseJudge:
    name: str
    def judge(self, prompt: str, response: str, probe_category: ProbeCategory | None = None) -> JudgementResult: ...

@dataclass
class JudgementResult:
    verdict: Verdict          # Verdict.REFUSED | PARTIAL | COMPLIED
    confidence: float         # [0, 1]
    judge_name: str
    explanation: str
    metadata: dict[str, Any]

class Verdict(str, Enum):
    REFUSED = "refused"
    PARTIAL = "partial"
    COMPLIED = "complied"
```

`realtime-safety-monitor.ResponseMonitor` accepts any `BaseJudge`. `adaptive-redteam` scorers extend `BaseScorer` (not `BaseJudge`) but share the verdict vocabulary.

---

## Backend Interface

**Location:** `safety_probe.backends.base`

```python
class BaseBackend:
    def generate(self, prompt: str, config: GenerationConfig) -> str: ...

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    # ...
```

`adaptive-redteam.TargetModel` wraps any `BaseBackend` with a simplified single-call interface. Backends are interchangeable — switching from `OpenAIBackend` to `TransformersBackend` requires no changes to red-teaming logic.

---

## Run Output Schema

`adaptive-redteam` writes JSONL output files consumed by `realtime-safety-monitor` for replay and pattern analysis.

**`scores.jsonl` record format:**
```json
{
  "failure_mode": "sycophancy",
  "prompt": "...",
  "response": "...",
  "score": 0.82,
  "signals": ["capitulation: you're absolutely right"],
  "scorer_name": "sycophancy_rule",
  "iteration": 2,
  "mutator": "reframe",
  "seed_id": "syc-003"
}
```

`realtime-safety-monitor.PatternLibrary` reads this format from `results/runs/<category>/<timestamp>/scores.jsonl`. The schema is stable; adding fields is backward-compatible, removing or renaming fields is a breaking change.
