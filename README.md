# ml-evolve

> A domain-agnostic, OpenEvolve-style evolutionary optimizer for ML algorithms,
> driven by Claude Code as the mutation agent. The skill body is fixed;
> everything scenario-specific arrives via a single `task_spec.yaml`.

`ml-evolve` runs an **architecture × parameter** search loop over your code:

- **Architecture search** is performed by Claude (web-research-grounded
  mutations to the `EVOLVE` block of a candidate program).
- **Parameter search** inside each architecture is performed by Optuna TPE.
- Candidates are organized into **islands** (research branches) with periodic
  re-planning to retire dead ends and inject new directions.
- Promotion gates move winners from cheap stages (e.g. `small`) to expensive
  ones (`medium`, `full`, `final`), so compute is spent only on what survives.

---

## Quickstart

```bash
# 1. Install as a user-level Claude Code skill
git clone <this-repo> ~/.claude/skills/ml-evolve

# 2. In your project directory, drop in three files:
#    - task_spec.yaml      (copy from templates/task_spec.example.yaml)
#    - initial_program.py  (a baseline candidate, must expose an EVOLVE block + PARAM_SEARCH_SPACE)
#    - evaluator.py        (must export evaluate(candidate_path, stage) -> {"score": float, ...})

# 3. Ask Claude Code:
/ml-evolve
```

The skill validates the spec, initializes a run, asks the plan agent to write
`research_plan.md`, then iterates: **select → mutate → param-batch → (promote)**.

---

## Repo layout

| File | Purpose |
|---|---|
| `SKILL.md` | Fixed 8-step control loop. No domain knowledge. |
| `TASK_SPEC.md` | Schema for `task_spec.yaml` — the only place scenario info lives. |
| `METHODOLOGY.md` | Generic principles: islands, TPE saturation, replan, promote. |
| `FEATURES_AND_ACCELERATION.md` | Generic compute / parallelism guidance. |
| `templates/task_spec.example.yaml` | Copy-paste starting point. |
| `scripts/evolve.py` | The runner: `init`, `plan`, `select`, `param-batch`, `promote`, `report`, `run-loop`. |
| `scripts/openevolve_local.py` | Environment checks and local OpenEvolve glue. |

---

## How the loop works

```
┌───────────────────────────────────────────────────────────────┐
│  evolve.py init     →  state.json, islands, candidate skeleton │
│  evolve.py plan     →  plan_agent_request.md  →  Claude writes │
│                         research_plan.md (one branch / island) │
│  ┌──────────────  loop  ──────────────┐                        │
│  │ evolve.py select    →  mutation_request.md                  │
│  │ (Claude edits EVOLVE block of next_candidate.py)            │
│  │ evolve.py param-batch  →  Optuna TPE × N trials on `small`  │
│  │ every K rounds:                                             │
│  │   evolve.py replan      →  retire/refresh/keep per island   │
│  │   evolve.py promote     →  re-evaluate top-K on `medium`+   │
│  └─────────────────────────────────────┘                       │
│  Stop on iterations / target_score / patience                  │
└───────────────────────────────────────────────────────────────┘
```

Every prompt to Claude is a **markdown file** written to the run directory
(`plan_agent_request.md`, `mutation_request.md`). This makes every step
auditable — you can re-read exactly what context drove each mutation.

---

## Case study: two-tower retrieval (run `recall-r3i9t8`)

A real run of `ml-evolve` on a sequential two-tower retrieval task, evolving
the **user encoder × loss × negative sampling × graph augmentation** space.

### Task spec (excerpt)

```yaml
objective:
  primary: HR@20
  direction: maximize

branches:                       # 3 islands, 3 independent hypotheses
  - name: loss_and_negatives
    hypothesis: "Tune InfoNCE/focal-InfoNCE/sampled-softmax × negative-mining policy"
    kill_criteria: "no improvement in 3 architectures"
  - name: sequence_encoder
    hypothesis: "SASRec/BERT4Rec-style attentive user tower, vary depth × heads × pooling"
  - name: graph_or_multi_interest
    hypothesis: "LightGCN / MIND / ComiRec, optionally with DeepWalk warm-start"

stages:                         # cheap → expensive
  search:  small                # 2 epoch, 100K events  (minutes)
  promote: medium               # 3 epoch, 200K events  (tens of minutes)
  final:   final                # test split, reporting

search:
  param_trials: 8               # TPE trials per architecture
  tpe_startup_trials: 3
  population_size: 40
  archive_size: 20

budget:
  iterations: 9
```

### What happened

| Generation | Event | Notable |
|---|---|---|
| 0 | `init`, plan agent writes 3 branches grounded in 2024–2025 papers + Kaggle writeups | — |
| 1 | First mutation on each island; baseline TwoTower @ HR@20 ≈ 0.087 | |
| 3 | branch_2 (graph) finds a LightGCN + focal-InfoNCE combo: **0.1336** | jump |
| 6 | TPE on branch_2's best architecture: **0.1353** | new high |
| 7 | branch_0 (loss) catches up: **0.1315** | |
| 8 | branch_1 (sequence) breakthrough: **0.1350** | 3-way tie |
| 9 | TPE convergence on all 3 islands; budget exhausted | stop |

### Final leaderboard (top 5, from `leaderboard.md`)

| rank | id | island | family | score (HR@20) | HR@10 | HR@50 |
|---:|---|---:|---|---:|---:|---:|
| 1 | `g0006_598773b0…` | 2 | branch_2 (graph) | **0.13534** | 0.1060 | 0.1701 |
| 2 | `g0009_b74ba5f7…` | 2 | branch_2 | 0.13529 | 0.1076 | 0.1704 |
| 3 | `g0009_48b2c26c…` | 2 | branch_2 | 0.13527 | 0.1078 | 0.1704 |
| 4 | `g0009_f00c8598…` | 2 | branch_2 | 0.13515 | 0.1072 | 0.1706 |
| 5 | `g0008_80048670…` | 1 | branch_1 (seq) | 0.13503 | 0.1101 | 0.1613 |

**Net result**: HR@20 improved from **0.087 → 0.1353** (+55%) in 9 generations
on the `small` stage, evaluating **54 candidates total across 3 parallel
research branches**. Compute footprint per run on a single T4 GPU: ~3 hours.

### Anatomy of one mutation (gen 6, branch_2)

The `mutation_request.md` Claude received contained:

- parent's full `PARAM_SEARCH_SPACE` and metrics
- branch_health JSON: `{trials: 8, saturated: true, slope: -0.0002}` → signal
  to make a **structural** change, not a parameter tweak
- the relevant slice of `research_plan.md` (LightGCN branch)
- the previous TPE batch's `best_params`, showing learning rate pegged at the
  upper edge of `[1e-4, 3e-3]` → hint to widen the range

Claude's mutation: switched edge-aggregation from sum to `mean`, added
DeepWalk warm-start init for item embeddings, widened `lr` to `[1e-4, 5e-3]`,
kept `PARAM_SEARCH_SPACE` size at 5 dimensions. TPE then found
`lr ≈ 1.5e-3, gcn_layers=3, temperature=0.04` → score 0.1353.

### Reproducing this case

```bash
cd examples/two-tower-retrieval/
python3 -m pip install -r requirements.txt    # torch, optuna, pandas, scipy
EVOLVE_PROJECT_DIR=$(pwd) \
  python3 ~/.claude/skills/ml-evolve/scripts/evolve.py \
  --run my-recall init --num-islands 3
# Then ask Claude Code:  /ml-evolve
```

---

## How ml-evolve differs from "auto" approaches

A natural question: *isn't this just AutoML / AutoGPT / evolutionary NAS?*  
ml-evolve occupies a distinct niche that none of these cover alone.

### vs. AutoML frameworks (AutoGluon, auto-sklearn, H2O)

| Dimension | AutoML frameworks | ml-evolve |
|---|---|---|
| **Search space** | Fixed model zoo (RF, XGB, MLP, …). Cannot invent architectures. | Claude proposes **novel architectures** grounded in recent papers — not from a pre-defined gallery. |
| **Optimization level** | Model selection + HPO only. | **Two-level**: Claude mutates the architecture; Optuna TPE tunes its parameters. |
| **Compute strategy** | Train all candidates (often ensemble). | **Stage promotion**: cheap → expensive — only winners advance. |
| **Diversity** | Ensembling for final model. | **Island branching** during search — prevents premature convergence on multimodal loss landscapes. |
| **Problem scope** | Tabular / CV / NLP with well-known families. | **Any domain** with a scalar evaluator: retrieval, ranking, alpha factors, RL policies, prompt programs, schedulers. |

**Bottom line**: AutoML picks from what exists; ml-evolve **invents what doesn't**.

### vs. General-purpose coding agents (AutoGPT, Claude Code, etc.)

| Dimension | Coding agents | ml-evolve |
|---|---|---|
| **Mutation scope** | Entire codebase — high risk of breakage. | **Evolve-only**: mutations are confined to the `EVOLVE` block. Everything else is frozen. |
| **Parameter tuning** | None (or ad-hoc). | **TPE batch** runs tens of structured trials after each mutation — separating architecture from parameter search. |
| **Evaluation rigor** | Single run, single split. | **Staged, gated evaluation**: cheap proxy → medium validation → final test. No data leakage. |
| **Search strategy** | Reactive ("improve this code"). | **Research plan** — each island has a hypothesis, kill criteria, and a grounded trajectory from papers / leaderboards. |
| **Reproducibility** | Hard — each run depends on LLM state. | **Every mutation is a file**: `mutation_request.md` captures the exact context that drove the edit. Fully auditable. |

**Bottom line**: Coding agents act, then see if it works. ml-evolve **plans, mutates, tunes, and gates** — a structured optimization loop, not a single-shot generation.

### vs. Traditional evolutionary algorithms (NEAT, Deep GA, regularized evolution)

| Dimension | Traditional EA | ml-evolve |
|---|---|---|
| **Mutation operator** | Hand-crafted (add/remove node, perturb weight). | **LLM-driven** — semantically aware mutations that understand the algorithm's logic. |
| **Parameter crossover** | Often coupled with architectural mutation. | **Decoupled**: Claude mutates structure; TPE handles parameters in a dedicated inner loop. |
| **Domain knowledge** | None — blind mutation. | **Plan agent** researches papers and writes grounded hypotheses per branch. |
| **Search landscape** | Single population. | **Multi-island** with retire/refresh — independent branches explore competing ideas in parallel. |

**Bottom line**: Traditional EA mutates blindly; ml-evolve **mutates with understanding**.

---

## How ml-evolve improves on Karpathy's AutoResearch

[AutoResearch](https://github.com/karpathy/autoresearch) (March 2026) proved that an LLM agent can autonomously iterate on ML training code overnight. ml-evolve takes that same core insight — "let the LLM drive experimental iteration" — and generalizes it into a production-grade optimizer with several structural improvements:

| Where AutoResearch stops | How ml-evolve improves |
|---|---|
| **Single domain**: only LLM pre-training (nanoGPT, `val_bpb`). | **Any domain**: one `task_spec.yaml` adapts the same loop to retrieval, ranking, alpha factors, RL policies, prompt programs — anything with a scalar evaluator. |
| **No parameter search**: the agent tweaks hyperparameters ad-hoc in code, no structured optimizer. | **TPE inner loop**: after each architecture mutation, Optuna TPE runs a dedicated parameter search — decoupling "what to change" from "how to tune it" doubles the optimization power per experiment. |
| **Single-threaded greedy**: one agent, one file, accept-or-rollback. Dead ends waste experiments. | **Multi-island branching**: independent branches explore competing hypotheses in parallel, with retire/refresh gates that prune dead ends and inject fresh directions. |
| **Flat experiment cost**: 5-minute wall-clock per run, regardless of quality. No progressive filtering. | **Staged promotion**: cheap (`small`) → medium → expensive (`final`). Only candidates that win at each stage consume more compute. Same total budget finds better solutions. |
| **Full-file mutation**: agent rewrites `train.py` anywhere — high breakage risk, hard to review diffs. | **Constrained EVOLVE block**: mutations are confined to the `EVOLVE` block; frozen code is protected. Diffs are small, targeted, and reviewable. |
| **Human writes strategy**: `program.md` is a free-text strategy document — no structure, no validation. | **Human writes spec, plan agent writes strategy**: `task_spec.yaml` is a validated schema; the plan agent researches papers and generates grounded hypotheses per branch autonomously. |
| **Experiment log only**: hard to replay or audit why a specific change was made. | **Every prompt is a file**: `mutation_request.md`, `plan_agent_request.md` capture exact context driving each decision. Full trajectory is re-playable. |

**Key improvement**: AutoResearch proved the *concept* of LLM-driven experimental iteration. ml-evolve turns it into a **structured optimization engine** — replacing ad-hoc exploration with TPE parameter search, multi-island diversity, staged compute gating, and an extension point (`task_spec.yaml`) that makes the same loop work across any ML domain.

---

## How ml-evolve improves on DeepMind's AlphaEvolve

[AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (May 2025) pioneered the idea of using an LLM ensemble as an evolutionary operator for algorithm discovery. ml-evolve is directly inspired by this paradigm, but makes three targeted improvements:

### 1. Decoupled search architecture (LLM → structure, TPE → parameters)

AlphaEvolve treats Gemini as the *only* evolutionary operator — architecture and parameter changes happen simultaneously in a single LLM-generated diff. This is elegant but conflates two fundamentally different search problems.

```
AlphaEvolve:       LLM ───────→ (architecture + parameters) in one shot

ml-evolve:  Claude mutates architecture ──→ TPE optimizes parameters
                 (semantic, paper-grounded)      (Bayesian, sample-efficient)
```

**Improvement**: By separating these, ml-evolve lets each search level use the right tool. Claude's strength is understanding algorithm semantics and proposing structurally sound changes. TPE's strength is sample-efficient numerical optimization — finding the best learning rate, depth, or temperature for a given architecture. They don't compete; they compose.

### 2. From Google-scale infrastructure to single-GPU accessibility

| Where AlphaEvolve requires | How ml-evolve improves |
|---|---|
| Proprietary Gemini models (Flash + Pro) with massive API quota. | Works with **any code-editing LLM** — Claude, GPT, open-source models. You bring the model you already use. |
| Distributed evaluation infrastructure for massive parallelism. | **Single-GPU / laptop** — evaluator runs locally. Stage promotion substitutes parallelism with progressive filtering. |
| DeepMind's internal program database (MAP-Elites grid). | **Simple island model** in local files — one `research_plan.md` per branch. Easy to inspect, modify, or restart. |
| 4-component distributed async system. | **Single `evolve.py` runner** (~76 KB) + one helper script. Read the whole thing in one sitting. |
| Not user-installable. | **Fully open-source**: `pip install` dependencies, write `task_spec.yaml`, run. |

**Improvement**: ml-evolve democratizes the AlphaEvolve paradigm — any researcher with a GPU and an LLM API key can run evolutionary algorithm optimization, without needing Google-scale infrastructure.

### 3. Safety and auditability by design

AlphaEvolve generates full-program diffs guided by natural-language context. Powerful, but risky — a bad diff can silently corrupt unrelated logic. ml-evolve constrains mutations to the `EVOLVE` block, protecting frozen code. Every mutation request is written to disk as a markdown file — you can re-read exactly what context drove each decision.

**Improvement**: Lower risk per mutation, higher visibility per decision. ml-evolve trades some of AlphaEvolve's raw mutation freedom for **safety, reproducibility, and inspectability** — qualities that matter more in production optimization than in open-ended scientific discovery.

### When each is suited

| Goal | Best fit |
|---|---|
| Solve open math problems / discover novel algorithms from scratch | AlphaEvolve (if you have Google-scale resources) |
| Optimize an existing ML algorithm for a domain-specific metric on your own hardware | **ml-evolve** |
| Run overnight LLM training experiments with minimal setup | Karpathy's AutoResearch |
| Production-grade evolution with auditable trajectory | **ml-evolve** |

---

## Why this design

- **Skill body is fixed.** No tuning advice in `SKILL.md`. All scenario
  knowledge lives in `task_spec.yaml`, so the same skill drives retrieval,
  ranking, tabular, RL, prompt-program, or scheduler problems.
- **Prompts are files, not API calls.** Every mutation request is written to
  disk so the trajectory is fully auditable and reproducible.
- **Islands prevent premature convergence.** Three independent branches with
  retire/refresh gates beat a single hill-climber on multimodal landscapes.
- **TPE handles parameters; Claude handles structure.** The two search levels
  don't fight each other.
- **Stage promotion controls compute.** You only pay `medium`/`full` cost on
  what already won at `small`.

---

## When *not* to use ml-evolve

- Single-knob hyperparameter tuning — use Optuna directly.
- Problems with no clear scalar evaluator — define one first.
- Tasks where one full evaluation takes >1 hour — promotion gates assume
  `small` stage is cheap.

---

## License

See `LICENSE`. Underlying `openevolve` runtime: Apache-2.0.

## Acknowledgements

Built on top of [OpenEvolve](https://github.com/codelion/openevolve), which
itself is inspired by DeepMind's AlphaEvolve. Mutation prompts are designed to
work with Claude (Anthropic) but any code-editing agent that can read a
markdown request and edit a Python file will work.
