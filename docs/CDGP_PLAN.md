# CDGP Execution Plan (Cross-Dataset Generalization Protocol)

## 1) Goal and Core Claim

Primary goal: prove that domain-guided active learning can adapt an agricultural vision model to a new farm/domain with far fewer labeled samples than random or standard uncertainty-only querying.

Core claim for publication: a calibrated transfer protocol can reduce local annotation demand while preserving or improving downstream detection/classification quality under realistic domain shift.

## 2) Scientific Questions and Hypotheses

1. RQ1: Does domain-guided querying outperform random and entropy-only querying at low labeling budgets?
Hypothesis H1: at matched budget, domain-guided AL yields higher target-domain performance.

2. RQ2: Does source pretraining accelerate adaptation on a different agricultural target domain?
Hypothesis H2: source-pretrained backbones reach a target metric threshold in fewer AL rounds than scratch.

3. RQ3: Is the adaptation robust to realistic deployment shift (illumination, background, sensor style)?
Hypothesis H3: calibration-stage domain-guided selection reduces performance drop under shifted target data.

## 3) CDGP Experimental Design (CVP-Aligned)

### Phase I: Toy Domain-Shift Verification

Dataset pair: MNIST -> MNIST-M (or equivalent synthetic style shift).

Objective: validate domain-guided query logic independent of agricultural noise complexity.

Runs:
- Random query baseline.
- Entropy-only baseline.
- Domain-guided AL (discriminator/confusion-based query score).

Deliverables:
- Adaptation curve (target accuracy vs labeled target samples).
- Mean +- CI across multiple seeds.
- Ablation of domain-guided scoring weight.

Exit criterion:
- Domain-guided AL statistically better than random at low budget.

### Phase II: Clean Agricultural Transfer

Primary pair (target): MinneApple -> WGISD (if available in your environment).
Fallback pair: any clean source->target agricultural pair with documented ontology mapping.

Objective: quantify cross-dataset transfer efficiency under AL.

Runs:
- Scratch on target + random AL.
- Scratch on target + entropy AL.
- Source-pretrained + entropy AL.
- Source-pretrained + domain-guided AL (CDGP main method).

Metrics:
- Detection: mAP50 (and mAP50-95 if feasible).
- AL efficiency: area under budget-performance curve.
- Transfer Efficiency Ratio (TER): speed/quality gain from source pretraining.

Exit criterion:
- Positive TER and significant low-budget gain for CDGP over baselines.

### Phase III: Wild Calibration Protocol

Scenario: unseen target orchard block or held-out shifted subset.

Protocol:
- Global scan with source-pretrained model.
- Select top-K most domain-confusing samples for rapid human annotation.
- Fine-tune and re-evaluate on full mission set.

Comparisons:
- No calibration.
- Random calibration set (same K).
- CDGP calibration set (same K).

Metrics:
- Delta in mAP/F1 after calibration.
- Label-efficiency gain per annotation minute.

Exit criterion:
- CDGP calibration significantly outperforms random calibration at same K.

## 4) Implementation Workstreams in This Repository

1. Add domain-guided query strategy plugin in `src/edge_al_pipeline/strategies/`.
2. Add model-side domain confusion score export in runner outputs and candidate metadata.
3. Add CDGP experiment runner under `src/edge_al_pipeline/experiments/`.
4. Add config files for CDGP toy/clean/wild runs under `configs/`.
5. Extend evaluation with TER and budget-efficiency reports under `src/edge_al_pipeline/evaluation/`.
6. Add tests for strategy logic, TER math, and report generation under `tests/`.

## 4A) Plug-and-Play Backbone and Strategy Architecture

Objective: make backbone and AL strategy switching configurable, reproducible, and low-friction.

### Backbone plug-and-play targets

1. Introduce a backbone registry keyed by config name.
2. Support at least these interchangeable backbones for controlled experiments:
- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `resnet18`
- `resnet50`
3. Keep a stable runner contract:
- `train_round(...)`
- `score_unlabeled(...)`
- `export_backbone(...)`
4. Expose backbone choice in config without code edits.

### AL strategy plug-and-play targets

1. Keep strategy builder/registry as the single dispatch point.
2. Enforce a common strategy interface:
- `select(candidates, k, seed=None) -> list[SelectionCandidate]`
3. Add CDGP domain-guided strategy as a first-class strategy name in config.
4. Preserve existing strategies (`random`, `entropy`, `k_center_greedy`) unchanged for baseline comparability.

### Config contract updates

Add explicit model switching fields (example):
- `model_name`: task-level runner
- `model_params.backbone_name`: selected backbone
- `strategy_name`: selected AL strategy
- `strategy_params`: strategy-specific hyperparameters

### Validation and tests

1. Config validation must fail fast on unknown backbone/strategy names.
2. Add tests that parameterize experiments over multiple backbone-strategy combinations.
3. Require smoke pass for each newly added backbone and strategy before large runs.

## 5) Statistical and Reporting Standards

- Use fixed seed set (>= 3 for pilot, >= 5 for submission runs).
- Report mean, standard deviation, and 95% CI for key deltas.
- Predefine primary endpoint per phase before final runs.
- Preserve all negative results and failure modes in supplement.

## 6) Timeline (Pressure-Resistant)

Week 1:
- Implement CDGP strategy + unit tests.
- Run toy pilot and sanity checks.

Week 2:
- Run full toy experiments and finalize Phase I plots/tables.
- Integrate clean transfer configs and TER evaluator.

Week 3:
- Run clean agricultural transfer experiments.
- Perform ablations (query weight, K, seeds).

Week 4:
- Run wild calibration experiments.
- Freeze figures, write methods/results/discussion.
- Final reproducibility check and submission package.

## 7) Manuscript Skeleton (Target: Agriculture-Focused Venue)

1. Introduction: domain shift as scaling bottleneck.
2. Related Work: AL in agri CV, transfer learning, domain adaptation.
3. Method: CDGP and query scoring.
4. Protocol: CVP with toy/clean/wild gates.
5. Results: main comparisons + ablations + negative results.
6. Deployment implications and limits.
7. Conclusion.

## 8) Next Prompt Handoff Template

Use this snippet in your next chat to continue without losing context:

```text
Continue CDGP execution from docs/CDGP_PLAN.md and docs/CDGP_COLAB_KAGGLE_SETUP.md.
Priority now: Week 1 implementation (domain_guided strategy plugin, config wiring for backbone_name and strategy_name plug-and-play, and matching unit tests).
Do not redesign the full repo; make minimal compatible changes with current Phase1-Phase3/GateA-GateC structure.
```

---

# Idempotent Task Checklist (Run-Safe)

Use this checklist to avoid duplicate work and keep reruns safe.

## A) Experiment Hygiene

- [ ] Fix and record seed list once in config files.
- [ ] Use stable experiment names per phase (do not rename mid-study).
- [ ] Keep dataset paths and versions frozen for a given run batch.
- [ ] Confirm each run writes `config_snapshot.json` and `splits.json`.

## B) Safe Execution Rules

- [ ] Before launching a run, check whether final report artifact already exists for that exact config+seed.
- [ ] If artifact exists and config hash matches, skip rerun.
- [ ] If config changed, create a new experiment name suffix (for example `_v2`) instead of overwriting interpretation.
- [ ] Never delete prior run directories during active paper drafting.

## C) Per-Phase Run Checklist

### Phase I (Toy)

- [ ] Run baseline random.
- [ ] Run baseline entropy.
- [ ] Run CDGP domain-guided.
- [ ] Generate paired comparison table and CI.
- [ ] Verify all seeds completed.

### Phase II (Clean Agri)

- [ ] Run scratch+random.
- [ ] Run scratch+entropy.
- [ ] Run source-pretrained+entropy.
- [ ] Run source-pretrained+CDGP.
- [ ] Compute TER and budget-efficiency summary.

### Phase III (Wild Calibration)

- [ ] Evaluate no-calibration baseline.
- [ ] Evaluate random calibration (K fixed).
- [ ] Evaluate CDGP calibration (same K).
- [ ] Report post-calibration gain and label-efficiency.

## D) Reproducibility and Paper Freeze

- [ ] Copy final figure/table inputs from run artifacts only (no manual edits).
- [ ] Keep one markdown changelog of any config change after first full run.
- [ ] Re-run a minimal smoke subset before submission to verify no regressions.
- [ ] Archive exact commit ID and commands used for camera-ready package.

## E) Plug-and-Play and Cloud Readiness

- [ ] Every major run logs both `model_name` and `model_params.backbone_name` in `config_snapshot.json`.
- [ ] Every major run logs `strategy_name` and `strategy_params` in `config_snapshot.json`.
- [ ] Add/verify one smoke config per new backbone.
- [ ] Add/verify one smoke config per new strategy.
- [ ] Keep cloud-specific config overrides in separate files instead of editing base configs.
