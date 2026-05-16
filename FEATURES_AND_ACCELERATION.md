# Acceleration and Parallelism (generic, spec-driven)

The skill body and scripts are domain-agnostic. Acceleration choices flow
from `compute.*` and `search.*` in `task_spec.yaml`, not from this document.

## 1. Where features live
Two patterns, declared in the spec via `candidate.allowed_mutations` and
`candidate.frozen`:

- **Evaluator-side**: evaluator pre-computes a frozen view; candidates work
  on top. Faster, deterministic, leakage-proof. Mark `feature pipeline` in
  `candidate.frozen`.
- **Candidate-side**: candidate builds features inside the training window,
  subject to evaluator-level leakage checks. Add `feature pipeline within
  training window` to `candidate.allowed_mutations`.

Hybrid forms are common; the spec decides.

## 2. Leakage rules (generic)
- Fit transforms on train only; apply to validation/test without refitting.
- Use time-aware or group-aware splits as appropriate; declare which under
  `evaluator.data.split_strategy`.
- The evaluator should emit `invalid: 1.0` on any detected leakage so the
  framework filters the candidate out of parent selection. Enumerate the
  detection rules under `evaluator.invalid_conditions`.

## 3. Stage budget
`stages.definitions[].cost_hint` is informative only. Typical shapes:

- inner search on `stages.search`: 100–1000 evaluations;
- promotions to `stages.promote`: 5–20 total;
- `stages.final`: 1–3 evaluations, reporting only.

`search.param_trials` should make one architecture cycle cost ~5–20× one
parameter trial. Too few trials → TPE underperforms; too many → the agent
cannot react to bad architectures. Tune this in the spec.

## 4. Optuna TPE batches
Driven by `search.param_trials` and `search.tpe_startup_trials`. The batch
writes one trial file per evaluation under `candidates/param_trials/`,
records metrics in `history.jsonl`, and updates the leaderboard
incrementally. The skill never changes these defaults internally — only the
spec does.

## 5. Hardware
- **CPU-only**: raise `compute.parallel_workers` toward physical core count;
  the evaluator must control BLAS threads to avoid contention.
- **Single GPU**: keep `compute.parallel_workers: 1` for the inner loop.
  Only raise it for `promote` if the evaluator multiplexes safely.
- **Multi-GPU**: the evaluator owns device dispatch (e.g. picks a free GPU
  per process). The framework does not schedule devices.
- **GPU enforcement**: if `compute.gpu_required: true`, the evaluator
  detects CPU fallback and emits `invalid: 1.0`. List that rule under
  `evaluator.invalid_conditions`.

`evolve.py torch-smoke` is available as a sanity helper before spending
a parameter batch on a torch candidate.

## 6. Promotion and early stopping
All driven by the spec:

- `promotion.every`, `promotion.top_k`, `promotion.parallel_workers`
- `budget.target_score`, `budget.patience`, `budget.min_improvement`

The skill applies these verbatim; it does not second-guess them.

## 7. Run hygiene
- One `run.name` per (problem, configuration). Set it in the spec or let the
  skill auto-generate.
- The user keeps `evaluator.py` under version control. Search results are
  only comparable across runs that share the same evaluator.
- Before changing the evaluator, archive `${EVOLVE_ROOT}/<run>/`.
