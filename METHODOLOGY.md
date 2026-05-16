# Methodology (generic, spec-driven)

Principles that apply to every run of `ml-evolve`, regardless of domain. None
of these principles encode task-specific advice — wherever a decision depends
on the problem, the answer lives in `task_spec.yaml`, not here.

## 1. One scalar objective, owned by the spec
A run maximizes exactly one scalar: `objective.score_key` from the spec
(direction `max`; if the spec sets `min`, the evaluator negates before
reporting). Multi-objective trade-offs are folded into the score formula or
into `objective.constraints` (which the evaluator turns into `invalid:1.0`).

## 2. The evaluator owns the protocol
Splits, purge/embargo, leakage guards, and the scoring formula are fixed for
the whole run. They are described in `evaluator.*` of the spec and
implemented in `evaluator.py`. Candidates never touch them.

Common protocol pitfalls (the spec should pre-empt these via
`evaluator.invalid_conditions` and `candidate.frozen`):

- features computed across the whole dataset before splitting (leakage);
- parent selection on the reporting stage (`stages.final`) — never do this;
- noisy validation: enlarge the set, average across seeds, or use a blocked
  statistic. The spec's `search.param_trials` should be tuned so TPE does not
  chase noise.

## 3. Staged search
Stages are labels, defined in `stages.definitions`. The inner loop runs on
`stages.search`; promotion gates on `stages.promote`; `stages.final` is
reporting-only and never feeds parent selection. The skill does not enforce
stage cost — that is evaluator business.

## 4. Architecture × parameters (two levels)
1. Plan / mutation agent proposes one architecture per `select`, inside the
   allowances of `candidate.allowed_mutations`.
2. `param-batch` runs `search.param_trials` Optuna TPE trials on that fixed
   architecture using `search_space`.
3. Only after the batch ends does the agent decide whether to keep, refine,
   or replace the architecture.

`apply_search_params(config, params)` may be implemented by the candidate for
custom param mapping. It is not the skill's concern how it's mapped.

## 5. Plan agent is constrained by the spec
At project start (and every `plan.replan_every` rounds if set), the plan
agent reads `task_spec.yaml`, the leaderboard, and `branch_health`, then
writes one branch per island into `research_plan.md`. The agent is
constrained to the slots declared in `spec.branches[]`: it may refresh a
branch's concept within that slot, but it may not invent a new slot, remove
a slot, or import a model family outside the slot's `model_family_hints`
(unless the user updates the spec).

## 6. Selection discipline
Defaults come from the spec:

- `search.exploitation_ratio` (default 0.65)
- `search.population_size` (default 40)
- `search.archive_size` (default 20)
- `search.elite_selection_ratio` (default 0.20)

If the search overfits the proxy: lower exploitation, raise archive. If it
wanders: do the opposite. Change these in the spec, not in the skill body.

## 7. Candidate boundaries
Candidates may modify only the EVOLVE block, and only along axes listed in
`candidate.allowed_mutations`. They must not touch anything in
`candidate.frozen`, the evaluator, the data split, or the score formula.
Network access and writes outside the run dir are forbidden unless the
evaluator explicitly enables them.

## 8. Reporting
After the search: `evolve.py report --stage <spec.stages.promote>`. The
report lists best candidate per branch (using `branches[].name`), the chosen
metric (`objective.name`), and run identity. Cross-stage / cross-task
comparisons are out of scope for the skill — the user collates those.
