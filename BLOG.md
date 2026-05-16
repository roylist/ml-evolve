# ml-evolve: You Built an ML Model. Now Watch It Improve Itself — A Self-Evolving Agent System for Algorithm Optimization

*How we built a self-evolving agent system that automatically researches, mutates, tunes, and improves ML algorithms — without a human in the loop. Think of it as an AI research assistant that works through the night, tests competing ideas in parallel, and leaves a complete audit trail of every decision.*

---

## 1. The problem: your model is only as good as your last architecture search

If you've ever trained a machine learning model, you've been here: you pick a loss function, choose an encoder architecture, set a few hyperparameters, train, evaluate, and repeat. Maybe you run a grid search or let Optuna sample a few hundred combinations.

This works — up to a point. Standard tools like Optuna, Hyperopt, or AutoML handle the continuous knobs (learning rate, dropout, batch size) and can even pick between pre-defined model families (Random Forest vs. XGBoost vs. MLP).

But there's a third dimension that none of these tools address: **structural innovation**. What if the best architecture for your problem isn't any of the standard options? What if you need a custom loss function that combines two existing ones? What if the latest paper from a top conference describes a technique that could boost your metric by 15% — but you'd need to implement and test it?

Historically, that's been a human job. You read the papers, formulate hypotheses, implement them, run experiments, keep what works, discard what doesn't. It's slow, expensive, and hard to scale across multiple competing ideas.

`ml-evolve` automates this loop. It's a **self-evolving agent system** that takes AlphaEvolve's evolutionary paradigm — treat algorithm search as code mutation — and rebuilds it as a production-ready framework for ML optimization. Instead of a single optimization algorithm, it orchestrates three specialized agents that collaborate to improve your model:

| Agent | Job description | When it works |
|---|---|---|
| **Plan Agent** | Reads papers, tech blogs, and leaderboards; writes a research plan with grounded hypotheses for each competing approach | Once at startup, then periodically to refresh stale directions |
| **Mutation Agent** (powered by Claude) | Edits the core algorithm — architecture, loss function, training logic — based on performance data and research plan | Once per generation, per competing branch |
| **Parameter Agent** (powered by Optuna TPE) | Runs Bayesian search over numerical parameters for each proposed architecture; reports when further tuning is pointless | Dozens of trials per architecture mutation |

This three-agent design puts LLM-driven structural evolution and TPE-driven numerical optimization into a single self-improving loop — auditable, resumable, and compute-aware.

---

## 2. What we optimized for

Before diving into the architecture, here's the design brief — the constraints that shaped every decision:

**Deployable in production, not just a notebook.** Every prompt the agent sees is written to disk as a file. State is serialized and resumable across machines. You can kill the process, restart on a different GPU, and pick up exactly where you left off.

**Works for any ML problem, not one domain.** The framework body contains zero domain knowledge. All task-specific info lives in a single YAML file. The same loop drives optimization for recommendation, ranking, tabular data, reinforcement learning, prompt engineering — anything with a scalar metric.

**Compute-aware by design.** Instead of running every candidate at full cost, the system uses multiple evaluation tiers (cheap proxy → medium validation → full test). Only candidates that win at each tier consume more compute. This is built in, not bolted on.

**Prevents premature convergence.** Multiple independent research branches explore competing hypotheses in parallel. Periodic replanning retires dead ends and injects fresh directions. The system deliberately avoids converging too early on the first plausible idea.

**Two-level search, one loop.** Architecture changes and parameter tuning are handled by different agents using different tools — no single optimization algorithm has to be good at both.

---

## 3. Architecture: Self-Evolving Agent System

The system operates as three coordinated agent layers, each with a distinct responsibility. Here's the full flow, from strategy to execution to feedback:

### Layer 1: Strategic Direction

| Component | Input | Output | Trigger |
|---|---|---|---|
| **Plan Agent** | Leaderboard snapshot, per-island branch health | `research_plan.md` with grounded hypotheses per island | Run init + periodic replan |

The Plan Agent reads recent papers and conference proceedings, then writes a research plan. Each island gets its own hypothesis (e.g., "does a transformer-based encoder outperform an MLP?"), kill criteria, and model family hints. The plan is a markdown file — editable, version-controllable, auditable.

### Layer 2: Parallel Evolution (×3 islands)

Each island runs an independent evolution loop. Islands are deliberately isolated — a dead end in one doesn't drag down the others.

| Island | Research branch | Mutation Agent | Parameter Agent |
|---|---|---|---|
| **Island 1** | Branch A hypothesis | Claude edits the EVOLVE block (architecture, loss, training logic) — grounded in cited papers | Optuna TPE runs N Bayesian trials; reports saturation when further tuning is futile |
| **Island 2** | Branch B hypothesis | Same mutation contract, different structural direction | Same TPE loop, independent parameter space |
| **Island 3** | Branch C hypothesis | Same mutation contract, different structural direction | Same TPE loop, independent parameter space |

Each iteration: **select a parent from the island's elite → mutation agent edits the candidate → parameter agent runs TPE trials → score → update leaderboard**.

### Layer 3: Evaluation & Advancement

Once candidates from all islands are scored:

| Component | Function |
|---|---|
| **Evaluator** | Runs the candidate on a fixed evaluation protocol (data splits, metric computation, seed averaging). Candidates never touch this logic. |
| **Stage Gate** | `small` → `medium` → `full` → `final`. Only candidates that win at each stage consume more compute. Catches architectures that overfit to cheap evaluation early. |
| **Leaderboard + Archive** | Stores the best candidates per island with full evaluation history. Used by the Plan Agent for replan decisions. |

### Layer 4: Meta-Learning (Feedback Loop)

| Component | Input | Decision |
|---|---|---|
| **Replan** | `island_health`: best-score trend, TPE saturation, needs_new_branch flag | Per island: **KEEP** (sharpen hypothesis), **REFRESH** (plateau detected, new direction in same space), or **RETIRE & REPLACE** (dead end, inject new branch) |

This is the system's anti-collapse mechanism. Without it, all islands eventually converge to the same lineage. The replan step is where the Plan Agent revises *its own search strategy* — the meta-cognitive loop that makes this a self-evolving system, not just a genetic algorithm with extra steps.

### 3.1 The task spec contract

A user authors one YAML file. Excerpt:

```yaml
objective:
  primary: HR@20
  direction: maximize

branches:                       # one per island; the agent fills in details during plan
  - name: loss_and_negatives
    hypothesis: "InfoNCE / focal-InfoNCE / sampled-softmax × negative-mining policy"
    model_family_hints: [InfoNCE, focal_InfoNCE, sampled_softmax, SSM]
    kill_criteria: "3 consecutive architectures with no improvement"
  - name: sequence_encoder
    hypothesis: "Attention-based user tower (SASRec / BERT4Rec / Mamba)"
  - name: graph_or_multi_interest
    hypothesis: "LightGCN / PinSAGE / MIND / ComiRec"

stages:
  search:  small      # 2 epoch, 100K events, minutes
  promote: medium     # 3 epoch, 200K events, tens of minutes
  final:   final      # test split, reporting

search:
  param_trials: 8
  tpe_startup_trials: 3
  population_size: 40
  archive_size: 20

compute:
  device: gpu
  parallel_workers: 1

budget:
  iterations: 9
  target_score: null
  patience: null

promotion:
  every: 3            # auto-promote every 3 steps
  top_k: 1

plan:
  allow_web: true
  replan_every: 5
```

The file is the **only** way information enters the loop. The skill explicitly refuses to inject domain knowledge from its own prompt.

### 3.2 The eight-step loop

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. status     →  verify python, evaluator.py, initial_program.py exist        │
│ 2. init       →  state.json, candidates/, islands, seed working candidate     │
│ 3. plan       →  plan_agent_request.md  →  agent writes research_plan.md       │
│ ╔══════════════════  per generation  ══════════════════╗                       │
│ ║ 4. select     →  pick parent, write mutation_request.md                   ║   │
│ ║ 5. mutate     →  agent edits EVOLVE block of next_candidate.py            ║   │
│ ║ 6. param-batch→  Optuna TPE × N trials at stage `small`                   ║   │
│ ║ 7. promote    →  every K steps, re-eval top-K at stage `medium`/`full`    ║   │
│ ║ 7'. replan    →  every M steps, retire/refresh/keep per island            ║   │
│ ╚════════════════════════════════════════════════════════════════════════════╝   │
│ 8. stop       →  iterations / target_score / patience trigger                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Why every prompt is a file

There's no hidden API call inside the loop. Instead, every piece of context the agent receives is written to disk as a markdown file:

- **`plan_agent_request.md`** — contains the leaderboard snapshot, per-island top scores, branch health metrics, and the current research plan. The plan agent reads this, does web research, and writes an updated `research_plan.md`.
- **`mutation_request.md`** — contains the parent's code path, its performance data, prior TPE saturation stats, and the relevant slice of the research plan. The mutation agent reads this and edits the candidate.
- **`history.jsonl`** — every evaluation result, every TPE trial, every promotion event. Append-only, machine-readable.

This design has two practical benefits:

1. **Full audit trail.** Six months from now, you can ask "why did we abandon branch_1?" and the answer is sitting in a markdown file in the run directory. You can replay the exact context that drove every decision.
2. **Engine independence.** The system doesn't care which LLM reads the files. Claude, GPT, an open-source model, or even a human manually editing the candidate — they all work without changing a line of orchestration code.

### 3.4 Islands and replan

Each island maintains its own family of candidates. The selector samples a parent from the island's elite + archive (configurable `exploitation_ratio`). Branch health is computed from:

- Best-score trend over the last K generations
- TPE saturation: did the last batch's `late_best - early_best` stay within `min_improvement`?
- `needs_new_branch` flag if the island's best is below 50% of the global best for N generations

When `replan_every` triggers, the plan agent is shown per-island health and must decide for each island:

- **KEEP** — sharpen hypothesis but keep concept
- **REFRESH** — plateau detected; replace hypothesis with a different direction in the same problem space
- **RETIRE & REPLACE** — proven dead end; introduce a structurally new branch

This is the framework's anti-collapse mechanism. Without it, all islands eventually converge to the global best lineage.

### 3.5 Two-level search

The mutation prompt makes the contract explicit:

> The candidate file **must** declare a module-level `PARAM_SEARCH_SPACE` dict. After your structural changes, **rewrite `PARAM_SEARCH_SPACE` so every dotted path resolves against the new `build_candidate()` config**.

The agent's job is to design the parametric family. Optuna's job is to find the best parameters within that family. Past TPE results are surfaced in the next mutation request:

> Previous TPE batch on this island: `{trials: 8, best_score: 0.135, slope: -0.0002, saturated: true}`
> High `trials` with `slope <= 0` and `saturated: true` → the prior structure is tapped out; prefer a **materially new** structural direction this iteration.

So saturation is the explicit signal for *when* to spend an architectural mutation vs. another parameter sweep.

### 3.6 Stage promotion

Five stages by convention: `smoke / small / medium / full / final`. The evaluator must dispatch on the `stage` argument and return a `score`. Candidates evaluated at `small` carry their score and metrics; when `promote` triggers, the top-K per island are re-evaluated at `medium`. Promoted rows land in the same `history.jsonl` with mode `promote_small_to_medium`, distinguishable from `tpe_param_trial` rows.

This is critical: a structural mutation that looks great at `small` may fail at `medium` because the small stage favors low-data regimes. Promotion gates surface that misalignment early and cheaply.

---

## 4. A real run: what this looks like in practice

Here’s what an actual run looks like. The task was a sequential ranking model — given a user’s history, predict which items they’ll interact with next. Three competing architectural families, 9 generations, single T4 GPU, about 3 hours.

| Generation | Best score | What happened |
|---:|---:|---|
| 1 | 0.087 | Baseline model — standard encoder with cross-entropy loss |
| 3 | 0.1336 | Branch 2 discovers a graph-enhanced encoder + modified loss combo — a **53% jump** |
| 6 | 0.1353 | TPE converges on branch 2's best architecture; saturation detected |
| 8 | 0.1350 | Branch 1 (attention-based encoder) catches up via a different structural path |
| 9 | 0.1353 | All three islands within 0.5% of each other; budget exhausted |

54 candidates evaluated total, ~3 hours compute, no human intervention. The mutation request that triggered the breakthrough at generation 6 contained:

- Saturation signal: branch health showed 8 TPE trials with near-zero improvement slope
- Prior TPE results: a key hyperparameter was pegged at the upper edge of the search range — hint to widen it
- Research plan excerpt suggesting a warm-start initialization technique

What the agent did in response: changed the aggregation method, added the warm-start technique, widened a hyperparameter range, and kept the search space compact. TPE then found the optimal numerical configuration for this new architecture.

This pattern repeats across runs: the saturation signal tells the agent when to stop tuning and start redesigning. The agent acts, TPE finds the new optimum, and the cycle continues.

---

## 5. How ml-evolve relates to AlphaEvolve

AlphaEvolve (DeepMind, 2024–2025) is the closest published system and the **direct intellectual ancestor** of ml-evolve. It is a code-evolution framework that uses Gemini to mutate programs guided by automated evaluators, and has produced novel results in matrix multiplication, data-center scheduling, and Google-internal algorithm improvements.

ml-evolve is the **first framework to take AlphaEvolve's evolutionary paradigm and rebuild it as a self-evolving agent system for ML tasks** — bringing multi-island evolution, agent-driven research, and TPE-accelerated parameter search out of Google-scale infrastructure and into a production-ready agent architecture. Where AlphaEvolve focuses on general-purpose algorithm discovery (sorting, math, formal verification), ml-evolve targets the specific needs of ML optimization: noisy scalar evaluators, expensive training pipelines, and the need for a self-improving agent loop that can be audited and deployed.

The shared lineage is real. Both systems:

- treat algorithm search as code mutation, not just hyperparameter optimization;
- score every candidate with a deterministic evaluator;
- maintain a population with diversity controls;
- iterate `propose → evaluate → select`.

The differences are where ml-evolve makes deliberate trade-offs:

### 5.1 What AlphaEvolve optimizes for

- **Scale**: Gemini's mutation throughput is enormous; AlphaEvolve runs thousands of evaluations per problem.
- **Closed loop**: orchestration, mutation, and evaluation are owned end-to-end by DeepMind infrastructure.
- **Reach**: it has been demonstrated on a wide variety of optimization problems including ones with formal verifiers (matrix multiplication kernels) and ones with noisy ML evaluators.

### 5.2 What ml-evolve optimizes for

ml-evolve redesigns the evolutionary paradigm as a **three-agent self-evolving system**, each agent optimized for a distinct role in the ML improvement loop:

- **Three-agent architecture**: a Plan Agent (research + strategy), a Mutation Agent (Claude, structural edits), and a Parameter Agent (Optuna TPE, numerical optimization). This decoupling lets each agent specialize — the Plan Agent stays grounded in recent literature, the Mutation Agent focuses on semantically meaningful code changes, and the Parameter Agent accelerates convergence through Bayesian search with explicit saturation detection.
- **Cost per run**: ml-evolve assumes a single workstation or a small cluster. A run is hours, not days; tens of candidates, not thousands. The two-level split (LLM = structure, TPE = parameters) is critical for this regime — you can't afford to spend LLM calls on hyperparameter sweeps.
- **Single agent, no proprietary infra**: the orchestration is a 1,800-line Python script and a markdown skill. The agent can be Claude Code, an API call, a human, or any tool that can read a markdown file and edit Python.
- **Auditability as a first-class concern**: every prompt is a file. AlphaEvolve's prompts are constructed inside its serving stack; for an external researcher reading a paper, they're a black box. For ml-evolve, the trajectory is on disk and reviewable line by line.
- **Explicit research grounding**: the mutation prompt mandates web search (Chinese & US tech-company blogs, conference papers, Kaggle writeups). Mutations are required to cite sources. AlphaEvolve uses Gemini's training-data priors implicitly; ml-evolve forces the agent to ground each mutation in recent external evidence.
- **Stage promotion baked in**: AlphaEvolve has progressive evaluation but it's a configuration choice. ml-evolve's task spec contract makes it a required field — you must declare a stage hierarchy. This nudges users toward compute-aware design.

### 5.3 Where ml-evolve is materially weaker

- No formal verification. AlphaEvolve produces verifiable improvements to matrix-multiplication algorithms; ml-evolve can't, because its loop assumes a noisy scalar evaluator.
- No mass parallelism. With `parallel_workers=1` (the default on a single GPU), the search is sequential.
- No automatic code minimization / certification of improvements. A winning candidate is a Python file, not a theorem.

The honest framing: ml-evolve is "AlphaEvolve redesigned as a self-evolving agent system for practical ML deployment — bringing multi-island evolution, agent-driven research, and TPE-accelerated parameter search to the case where an ML engineer wants to run autonomous algorithmic improvement on a real problem, and needs to explain every step to their team."

---

## 6. How ml-evolve relates to Karpathy's AutoResearch

[AutoResearch](https://github.com/karpathy/autoresearch) is Andrej Karpathy's recent project ("give an AI agent a small but real LLM training setup and let it experiment autonomously"). It is the closest spiritual relative to ml-evolve: both are file-based, agent-driven research loops designed to run autonomously for hours without supervision.

The critical distinction is that **ml-evolve inherits AlphaEvolve's multi-island evolutionary architecture**, while AutoResearch uses a single-stream greedy loop. This inheritance is the primary structural difference: ml-evolve maintains multiple independent research branches with periodic replan (KEEP / REFRESH / RETIRE & REPLACE), whereas AutoResearch follows one trajectory, accept-or-rollback. On multimodal ML landscapes — the norm for production ML problems — this difference is decisive.

Worth comparing in detail because the differences are deliberate engineering choices, not accidents.

### 6.1 AutoResearch in one paragraph

Three files: `prepare.py` (data, frozen), `train.py` (the single file the agent edits — full GPT model, Muon + AdamW optimizer, training loop), `program.md` (agent instructions and research directives). Each training cycle runs for exactly 5 wall-clock minutes (~12 experiments/hour). The agent reads `program.md`, edits `train.py`, trains, evaluates on **validation bits per byte** (`val_bpb`, vocabulary-size-independent), then decides keep or discard. The metric is the only arbiter; the agent is the only proposer; the loop is otherwise unconstrained.

### 6.2 The shared vision

Both ml-evolve and AutoResearch agree on:

- The agent is the **proposer**, not the orchestrator. A small Python harness runs the loop.
- Instructions live in **markdown files**, not in a hidden system prompt — the human programs the agent through readable, version-controllable text.
- A **frozen evaluation contract** (AutoResearch: `prepare.py` + `val_bpb`; ml-evolve: `evaluator.py` + `score`) prevents the agent from cheating its own metric.
- **Wall-clock economy** matters: experiments must be bounded so the agent can iterate many times overnight rather than burn a night on one run.
- **Reproducibility-by-disk**: experiments produce artifacts that survive the session.

### 6.3 Side-by-side

| Dimension | AutoResearch | ml-evolve |
|---|---|---|
| Domain | LLM pretraining on a fixed dataset | Domain-agnostic — retrieval, ranking, tabular, RL, prompt programs, schedulers |
| Editable surface | The entire `train.py` (model + optimizer + loop) | The `EVOLVE` block of `next_candidate.py` (model + algorithm); `evaluator.py` is frozen |
| Search topology | Single sequential stream | Multi-island population (typically 3) with archive |
| Diversity mechanism | Implicit (agent self-selects directions) | Explicit (islands + replan with KEEP / REFRESH / RETIRE & REPLACE per island) |
| Parameter vs. structural search | Conflated — agent proposes both | Decoupled: agent does structure, Optuna TPE does parameters within each structure |
| Per-trial compute budget | Fixed 5-minute wall clock | Stage hierarchy (`smoke / small / medium / full / final`) declared in task spec |
| Cost-aware promotion | None — every run is full 5 min | Promotion gates: top-K from `small` re-evaluated at `medium`/`full` |
| Decision unit | Keep-or-discard a single run | Population update with archive, elite selection ratio, exploitation ratio |
| Halt criteria | Run overnight / human kill | Explicit `iterations` / `target_score` / `patience` |
| State persistence | Git commits + run artifacts | `state.json` + `history.jsonl`, resumable across sessions/machines |
| Research grounding | Whatever `program.md` says | Mandatory web-research section in every mutation prompt, with named source categories and citation requirements |
| Metric | `val_bpb` (single scalar) | User-defined primary score, but every evaluation may report a full metric dict |
| Loop driver | Shell loop calling the agent | `evolve.py` subcommands (`init / plan / select / param-batch / promote / replan / run-loop`) |

### 6.4 What ml-evolve optimizes relative to AutoResearch

These are deliberate additions to handle failure modes that emerge when you run a single-stream, edit-everything loop on tasks more complex than nanoGPT pretraining. The most fundamental improvement is inheriting AlphaEvolve's **multi-island evolutionary architecture** — ml-evolve is not a single-stream loop, but a population-based optimizer with structured diversity management:

1. **Multi-island evolution (inherited from AlphaEvolve).** This is the primary structural difference. AutoResearch follows one trajectory with accept-or-rollback — a greedy hill climber. ml-evolve maintains multiple independent research branches with periodic replan (KEEP / REFRESH / RETIRE & REPLACE). On multimodal ML landscapes — the norm for production ML problems — this is decisive. In our test run, the winning architecture came from a branch the agent would not have explored under a single greedy stream — discovered only because multiple hypotheses were running in parallel.

2. **Parameter sweeps belong to TPE, not the agent.** On nanoGPT, learning-rate sweeps are cheap enough that an agent burning 12 experiments/hour can afford to grid-search by trial and error. On a realistic ML task where each evaluation takes minutes and involves multiple continuous hyperparameters, that approach doesn’t scale. ml-evolve's `PARAM_SEARCH_SPACE` contract pushes the LLM out of the loop for parameter search and lets Optuna TPE do what it's good at — 8 trials per architecture, started from prior elite values, with explicit saturation detection. The agent only gets called again when there's a *structural* decision to make.

3. **Saturation is a first-class signal, not an inference the agent has to make.** AutoResearch's agent sees its own run history but has no explicit telemetry telling it "your last 8 trials on this architecture had slope -0.0002, you should mutate structurally now." ml-evolve computes this from TPE's trial-by-trial best-score series and injects `{trials, slope, saturated, late_best - early_best}` into the next mutation prompt. The result: structural mutations happen when there's evidence they're needed, not on a fixed cadence.

4. **Stage hierarchy lets you spend compute where it matters.** AutoResearch's 5-minute budget is the *only* budget — if your real evaluator takes 30 minutes per epoch, the design doesn't translate. ml-evolve's task spec requires you to declare cheap-to-expensive stages and which stage drives the inner loop. Promotion re-evaluates winners at higher cost, so structural mutations that look good at `small` but degrade at `medium` are caught early.

5. **Web research is enforced per mutation, not delegated to the agent's discretion.** AutoResearch can ask for research in `program.md`, but it's a request. ml-evolve's mutation prompt has a *required* "Research Before Editing" section that names specific source categories (Chinese AI tech blogs, top-tier conferences, Kaggle writeups), demands 2–3 independent citations per structural decision, and rejects mutations that aren't grounded in 2024–2025 evidence. This is the single most important factor in mutations not drifting to the agent's training-data priors.

6. **Domain-agnostic by construction.** AutoResearch is purpose-built for LLM pretraining. To adapt it to a recommender task, you would rewrite `prepare.py`, `train.py`, the metric, and most of `program.md`. ml-evolve was designed so that swapping `task_spec.yaml`, `evaluator.py`, and `initial_program.py` is enough — the framework body never changes.

7. **Resumability across sessions and machines.** AutoResearch's state lives in git commits on `train.py` plus run logs. ml-evolve's lives in `state.json` (population, archive, TPE saturation, replan history) and `history.jsonl` (every evaluation ever). You can kill the run, restart on a different machine, and pick up exactly where you left off. This matters when overnight = "spot-instance overnight."

8. **Industrial-grade audit trail.** Every prompt is a file on disk — `mutation_request.md`, `plan_agent_request.md`, `research_plan.md`. AutoResearch keeps experiment logs; ml-evolve keeps the complete decision context, reviewable line by line. This is not a nice-to-have — it's required for deployment in production ML pipelines where every algorithmic decision must be explainable.

### 6.5 Where AutoResearch is stronger

- **Ergonomics on a well-posed task.** If your task fits "edit a single file, train for 5 minutes, check `val_bpb`," AutoResearch's three-file setup is genuinely lower-friction than authoring a `task_spec.yaml` with 8 sections and three branches.
- **No premature commitment to a search structure.** ml-evolve forces you to name your branches up front. If the search space is truly unknown, AutoResearch's freeform stream may find directions the spec author wouldn't have written down.
- **Real-time iteration speed on small models.** ~12 experiments/hour is a tighter feedback loop than ml-evolve's typical generation cadence (1–3 generations/hour with web research).
- **Designed for the LLM-pretraining domain it targets.** When the domain matches, every choice in AutoResearch is well-tuned for it; ml-evolve's generality has overhead.

The honest framing: **AutoResearch and ml-evolve are siblings, not competitors.** AutoResearch optimizes for the case where the human research question is "what is the best small GPT recipe under 5 minutes of compute?" — a single, well-bounded, single-stream search. ml-evolve inherits AlphaEvolve's **multi-island evolutionary architecture** and builds it into a **self-evolving agent system** with three specialized agents. It optimizes for "what is the best algorithm in this ML problem space, given a noisy evaluator, a compute budget I have to account for, multiple plausible architectural families, and a need to explain every decision to my team six months from now?" Move from AutoResearch to ml-evolve when the search space is multimodal, the evaluator is expensive enough that you can't afford to run it at full cost on every candidate, the problem domain is not LLM pre-training, or you need an audit trail that survives the experiment.

---

## 7. Summary: what we built and why it matters

ml-evolve is a **self-evolving agent system** for ML algorithm optimization — not an academic experiment, but a production-ready framework where three specialized agents collaborate to continuously improve your model.

The key design decisions:

1. **Two-level search** — an LLM handles structural mutations (architecture, loss function, training logic), while Bayesian optimization handles parameter tuning. Each uses the right tool, improving compute efficiency by roughly 10× over letting the LLM do everything.
2. **File-based audit trail** — every prompt is a file on disk. Every decision can be replayed and reviewed, six months later.
3. **Multi-island evolution** — inherited from AlphaEvolve. Multiple competing hypotheses evolve in parallel, with structured replanning to retire dead ends and inject fresh ideas.
4. **Compute gating** — cheap evaluation first, expensive evaluation only for winners. Built into the specification, not an optional feature.
5. **Research-grounded mutations** — the plan agent must read and cite recent sources. Mutations don't drift to the LLM's stale training data.
6. **Saturation-driven timing** — TPE telemetry tells the mutation agent when to stop tuning and start redesigning. Neither AlphaEvolve nor ad-hoc agent loops surface this signal.
7. **Domain-agnostic** — one framework, any ML problem. Swap the spec and the evaluator; the loop stays the same.
8. **Production-ready** — resumable across machines, no proprietary infrastructure, works with any code-editing LLM.

The result, in one real run: a 53% improvement in the primary metric, 54 candidates evaluated, ~3 hours on a single GPU. Not a benchmark — a practical demonstration of what a self-evolving agent system can deliver.

---

## 8. Limitations and future work

- **Verifier-free.** ml-evolve assumes a noisy scalar evaluator. It cannot certify correctness of evolved code (no formal verification, no test-suite-as-evaluator pattern). Adding a fixed regression test as a hard gate before scoring would be a natural extension.
- **Single-machine assumption.** Distributed runs require external orchestration; the current `parallel_workers` is for in-process TPE concurrency only.
- **Plan agent latency.** Web research per replan is the slowest step. Caching searches across runs (and across users for the same task spec) would help.
- **No cross-run learning.** Each run starts cold. A `priors/` directory of `mutation_request.md` + outcome pairs across past runs would let the agent learn from prior failures on this task.
- **Bench coverage.** The framework has been validated on retrieval, ranking, and a handful of tabular tasks. Coverage on RL, prompt-program, and scheduler tasks is theoretical, not yet measured.

---

## 9. Closing: self-evolving agents for production ML

ml-evolve is a **self-evolving agent system** — the first framework to bring AlphaEvolve's evolutionary paradigm into a closed-loop agent architecture designed for industrial ML deployment:

- three specialized agents (Plan, Mutation, Parameter) collaborate in a self-improving loop;
- multi-island evolution inherited from AlphaEvolve provides multimodal coverage and prevents premature convergence;
- agent-driven research grounds every mutation in current literature, not stale priors;
- TPE-accelerated parameter search with explicit saturation feedback tells each agent *when* to act;
- every prompt is a file — audit trail as a first-class requirement, not an afterthought.

It is not a replacement for AlphaEvolve at scale, nor for AutoResearch's tight five-minute loop on the nanoGPT-pretraining domain it targets. It is the right tool when you have a production ML optimization problem — a noisy scalar evaluator, a few hours of GPU, multiple plausible architectural families to compare, and a need for a self-evolving agent system that can explain every decision.

The framework is open-sourced under the `ml-evolve` skill. Contributions, alternative task specs, and case studies welcome.
