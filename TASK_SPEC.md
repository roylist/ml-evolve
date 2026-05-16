# task_spec.yaml — Standardized Input Contract

The `ml-evolve` skill body is fixed. Everything that depends on the user's
problem flows through one file: `task_spec.yaml` (YAML or JSON accepted).
This document is the authoritative schema. Anything not in this schema does
not belong in the skill body either; extend the schema instead.

## Top-level fields

```yaml
version: 1                       # schema version, integer, required
task_name: <string>              # short identifier used in run names

context:                         # free-text background, surfaced verbatim
  domain: <string>               # e.g. "two-tower retrieval", "alpha factor",
                                 # "tabular binary classifier", "RL policy" —
                                 # this string is opaque to the skill
  description: <string>          # 1–5 sentences of problem context
  references: [<url|path>, ...]  # optional links the plan agent may consult

objective:
  name: <string>                 # human-readable metric name
  score_key: <string>            # JSON key on evaluator stdout (default "score")
  direction: max | min           # framework always maximizes; if `min`, the
                                 # evaluator must negate before reporting
  constraints:                   # optional; evaluator enforces via `invalid:1.0`
    - name: <string>
      rule: <string>             # free text, evaluator interprets

evaluator:
  entrypoint: <path>             # default "evaluator.py" in project dir
  timeout_sec: <int>             # default 0 (no timeout)
  data:
    source: <string>             # path, URI, or table reference
    split_strategy: <string>     # e.g. "time-forward", "k-fold-on-train",
                                 # "group-by-user" — opaque to skill
    purge_embargo: <string|null>
  invalid_conditions:            # surfaced to evaluator as guidance
    - <string>
  notes: <string>                # any additional contract the evaluator owns

candidate:
  template: <path>               # default "initial_program.py"
  allowed_mutations:             # whitelist of code regions / aspects
    - <string>                   # e.g. "loss", "sampler", "tower encoder",
                                 # "feature pipeline" — the plan/mutation
                                 # agent uses ONLY these labels
  frozen:                        # things candidates MUST NOT change
    - <string>                   # e.g. "evaluator", "data split", "score formula"

search_space:                    # optional; mirrors PARAM_SEARCH_SPACE
  "<dotted.path>":
    type: float | int | categorical
    low: <number>                # float/int
    high: <number>
    step: <number|null>
    log: <bool>                  # float, log-scale
    choices: [<...>]             # categorical

branches:                        # one entry per research island
  - name: <string>               # used verbatim in reports
    hypothesis: <string>         # 1–3 sentences; the plan agent expands this
    model_family_hints: [<string>, ...]   # optional, narrows model class
    kill_criteria: <string>      # when to retire / refresh
  # Repeat for as many islands as desired. spec.branches.count is implied.

stages:
  search: <label>                # stage used in the inner loop (e.g. "small")
  promote: <label>               # stage used for promotion (e.g. "full")
  final: <label>                 # reporting-only stage (e.g. "final")
  definitions:                   # optional richer description per stage
    - name: <label>
      cost_hint: <string>        # "seconds" | "minutes" | "tens of minutes"
      sampling: <string>         # opaque description for evaluator
      purpose: <string>

search:
  param_trials: <int>            # --param-trials-per-architecture
  tpe_startup_trials: <int>      # default 5
  exploitation_ratio: <float>    # default 0.65
  population_size: <int>         # default 40
  archive_size: <int>            # default 20
  elite_selection_ratio: <float> # default 0.20

compute:
  device: cpu | gpu | mixed
  gpu_required: <bool>           # evaluator should emit invalid:1 on CPU fallback
  parallel_workers: <int>        # default 1
  notes: <string>

budget:
  iterations: <int>              # architecture cycles for run-loop
  target_score: <number|null>
  patience: <int|null>
  min_improvement: <float>       # default 1e-6

promotion:
  every: <int>                   # 0 disables auto-promotion
  top_k: <int>                   # per-island quota
  parallel_workers: <int>

plan:
  allow_web: <bool>              # plan agent may use web search
  replan_every: <int>            # 0 disables replanning

run:
  name: <string|null>            # optional run id; auto-generated if absent
  seed: <int>                    # default 42
  root: <path|null>              # overrides EVOLVE_ROOT

secrets:                         # values exposed to subprocesses via env only;
  - name: <ENV_VAR_NAME>         # never written to logs, never echoed
    description: <string>
```

## Validation rules

The skill rejects the spec and asks the user to fix it if any of:

- `objective.score_key` missing.
- `branches` empty.
- `stages.search` not present in `stages.definitions` (when `definitions` is
  given).
- Any field in `candidate.allowed_mutations` overlaps `candidate.frozen`.
- A `search_space` entry has both `choices` and `low/high`.

## Why the spec exists

The previous skill body sometimes leaked scenario specifics into examples
("when evolving a two-tower retriever, try …"). That coupling makes the skill
behave differently per user, which is exactly what the body must NOT do. Move
all such guidance into `branches[].hypothesis`, `context.description`, and
`candidate.allowed_mutations`. The skill then reads them and surfaces them to
the plan/mutation agent — without ever encoding the same advice into the
fixed loop.

## Minimal example

See [templates/task_spec.example.yaml](templates/task_spec.example.yaml) for a
copy-pasteable starting point. That file is the only place a new user needs
to edit; the skill body, scripts, and methodology docs stay untouched.
