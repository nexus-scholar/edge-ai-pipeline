## One-page implementation spec: Edge-Based Active Learning Pipeline (Agricultural CV)

**Source:** “Strategic Realignment of Edge-Based Active Learning…” 
**Objective:** Deliver a validated, edge-feasible Active Learning (AL) loop for agricultural computer vision by following a curriculum protocol: **Toy → Controlled Agri → High-quality Field**, avoiding early failure modes from noisy/cluttered field sets like PlantDoc. 

---

### 1) Scope (what we will build)

**In scope**

* A **reproducible AL training pipeline** with pluggable query strategies (Random, Entropy/Least-Confidence, Coreset/k-Center Greedy). 
* **Distributed AL (Edge Student + Fog/Cloud Teacher)**: edge pre-filters uncertain samples; teacher verifies informativeness to mitigate “outlier problem” from background clutter. 
* Edge profiling hooks (latency, memory, optional power/thermal) and **quantization experiments (FP32 vs INT8)** to test overconfidence effects on uncertainty. 

**Out of scope (v1)**

* Full production annotation platform (we’ll use a simple labeling interface or existing tools)
* Direct AL benchmarking on PlantDoc until the curriculum gates pass (data pathology confounds early AL validation). 

---

### 2) Acceptance criteria (Definition of Done)

**Gate A — Toy verification**

* On **Fashion-MNIST**, uncertainty sampling (Entropy) must beat Random by **>5% accuracy at 10% labeled budget**, averaged over multiple seeds with confidence intervals. 

**Gate B — Agri feature readiness**

* Train **MobileNetV3** on **PlantVillage**, export **Agri-pretrained backbone**, verify transfer improves convergence vs ImageNet init. 

**Gate C — Field proxy validation**

* On **WGISD**, AL must beat Random on detection (mAP vs labeled budget) and compare **classification vs localization uncertainty (box jitter/variance)**. 

**Edge feasibility**

* Provide a profiling report: pool-scan time, inference latency, memory footprint; include FP32 vs INT8 uncertainty behavior. 

---

### 3) System design (modules + responsibilities)

1. **Data Pool Manager**

   * Maintains splits: `L` (labeled), `U` (unlabeled), `V` (val), `T` (test)
   * Tracks dataset version/hash and split manifest for reproducibility.

2. **Model Runner**

   * Supports tasks: classification (CNN/MobileNetV3), detection (YOLO-nano class).
   * Produces predictions + uncertainty scores on `U`.

3. **Query Strategy Engine (plugin interface)**

   * `select(U, model, k, strategy_config) -> batch_ids`
   * Strategies (v1): Random, Entropy/Least-Confidence, k-Center Greedy (coreset). 

4. **Teacher Verifier (optional / Phase 3+)**

   * Re-scores candidate set from Edge Student, rejects clutter/outliers; outputs final query batch. 

5. **Trainer/Evaluator**

   * Retrain (or warm-start) each AL round; logs metrics per round and per seed; outputs learning curves.

6. **Edge Profiler**

   * Captures: inference latency/sample, pool scan time/round, memory use; optional thermal/power.
   * Runs FP32 + INT8 experiments. 

---

### 4) Data + experiment protocol (the curriculum)

**Phase 1: Fashion-MNIST (logic sanity check)**

* Model: small CNN
* Strategies: Random vs Entropy
* Output: learning curves; must pass Gate A. 

**Phase 1b: CIFAR-10 (RGB pipeline validation)**

* Verify correct RGB preprocessing and robustness to blur/low contrast proxies. 

**Phase 2: PlantVillage + Fruits-360 (controlled agriculture)**

* PlantVillage: train MobileNetV3, freeze/export backbone (“cold start” mitigation). 
* Fruits-360: simulate severe imbalance; verify strategy can “hunt” minority classes; add weighting if needed. 

**Phase 3: WGISD (high-quality field proxy)**

* Model: YOLOv8/YOLOv10 nano-class
* Compare uncertainty types: classification vs localization; run teacher verification variant.
* Include FP32 vs INT8 uncertainty sanity check. 

---

### 5) Interfaces, data formats, and logging (engineering contract)

**Config**

* Single experiment config file (YAML/JSON): dataset, model, strategy, `k`, rounds, seeds, quantization mode, teacher on/off.

**Artifacts**

* `splits.json` (ids for L/U/V/T)
* `round_{r}_selected.csv` (selected ids + scores + strategy metadata)
* `metrics.csv` (per round: accuracy/F1 or mAP; per seed)
* `profile.csv` (latency/mem; quant mode; device info)
* Model checkpoints per round (edge + teacher if used)

**Reproducibility**

* Fixed RNG seeds; deterministic flags where feasible; dataset hash/version recorded in every run. 

---

### 6) Non-functional requirements

* **Runtime:** pool scoring must be batchable; allow sub-sampling of `U` for profiling.
* **Compute constraints:** support quantized inference (INT8) and report impact on uncertainty calibration. 
* **Fail-fast checks:** if Entropy ≤ Random on Fashion-MNIST, block progression and open a bug ticket (code/HP issue, not data). 

---

### 7) Risks and mitigations (from the report’s post-mortem)

* **Background clutter ⇒ “outlier problem”**: mitigate with teacher verification + curated high-quality field proxy first (WGISD), not PlantDoc. 
* **Label noise breaks oracle assumption**: enforce label QA on field datasets; avoid noisy sets until pipeline validated. 
* **Edge bottlenecks**: measure scan latency and thermal; use duty-cycling and/or coreset/pruning to reduce compute. 

---

### 8) Deliverables checklist

* ✅ AL pipeline (toy + RGB) passing Gate A
* ✅ Agri-pretrained backbone exported (PlantVillage)
* ✅ WGISD AL report: uncertainty comparison + edge profiling + quantization effects
* ✅ Runbook: configs, artifacts, and how to reproduce plots and results

If you paste your target edge device (e.g., Raspberry Pi 4 vs Jetson Nano vs Coral), I can tighten the profiling thresholds (latency/memory) and recommend the exact deployment format (TFLite vs TorchScript) without changing the spec.
