# Refund Sentinel — Evaluation Methodology & Results

> This document records the benchmark actually produced by the current implementation. Results are reported as measured values from the persisted evaluation artifact, not as aspirational architecture claims.

## Table of Contents

## Table of Contents

- [Evaluation objective](#1-evaluation-objective)
- [Experimental design](#2-experimental-design)
- [Dataset construction](#3-dataset-construction)
- [Train / validation / held-out protocol](#4-train--validation--held-out-protocol)
- [Baselines](#5-baselines)
- [Operating thresholds](#6-operating-thresholds)
- [Final held-out results](#7-final-held-out-results)
- [Financial exposure results](#8-financial-exposure-results)
- [False-positive cost sensitivity](#9-false-positive-cost-sensitivity)
- [Per-scenario results](#10-per-scenario-results)
- [Interpretation](#11-interpretation)
- [What the benchmark supports](#12-what-the-benchmark-supports)
- [What the benchmark does not support](#13-what-the-benchmark-does-not-support)
- [Metric coverage and caveats](#14-metric-coverage-and-caveats)
- [Reproduction](#15-reproduction)
- [Reporting checklist](#16-reporting-checklist)


---

## 1. Evaluation objective

The evaluation asks whether combining refund-lifecycle, individual behavioral, structural and cross-account coordination signals provides a more useful defense against coordinated refund abuse than either individual-only or graph-only reasoning.

The benchmark is deliberately falsifiable:

```text
Baseline A → what individual behavior detects
Baseline B → what raw structural connectivity detects
Baseline C → what the full system detects
```

The comparison is especially important on legitimate lookalikes because the central architectural claim is not that graphs are useless; it is that **structure should be confirmed by behavior before it materially drives risk**.

---

## 2. Experimental design

The evaluator uses three partitions:

```text
TRAIN
  ↓
fit preprocessing + model

VALIDATION
  ↓
select operating thresholds

HELD-OUT TEST
  ↓
final reporting exactly once
```

Point-in-time feature construction is enabled so information occurring after the scored refund is not used to construct its features.

The final held-out benchmark is stored in:

```text
data/results.json
```

---

## 3. Dataset construction

The benchmark is simulator-generated and contains domain events rather than a bare `(customer_id, fraud)` table.

The simulator creates sequences such as:

```text
payment.created
payment.captured
order lifecycle
refund.requested
refund.created
refund.processed
```

and links related entities through supported structural identifiers.

### Current generated sizes reported by the scripts

| Partition | Events generated | Refund / evaluation rows reported by generator | Abuse | Legitimate |
|---|---:|---:|---:|---:|
| Train | 1,657 | 74 | 39 | 35 |
| Validation | 877 | 54 | 39 | 15 |
| Test generator output | 1,375 | 60 | 39 | 21 |
| Held-out evaluator rows | — | 72 | 39 | 33 |

The evaluator's held-out set contains the final rows used for comparative scoring.

---

## 4. Train / validation / held-out protocol

### Training

The model is fit only on training data.

### Validation

Validation data is used to select operating thresholds.

The selected thresholds are not chosen from held-out labels.

### Held-out test

The held-out partition is used for final comparative reporting.

The current protocol records:

```text
point_in_time_features = true
threshold_selection = validation only
held-out labels = never used for threshold selection
```

### Leakage controls

The simulator/evaluation design checks for:

- customer overlap between partitions;
- cluster overlap between partitions;
- held-out mechanism contamination;
- inaccessible labels during feature computation;
- post-label features.

A metric should not be reported if a leakage check fails.

---

## 5. Baselines

### Baseline A — Individual-only

Uses individual refund/customer behavioral features and excludes graph/cluster coordination evidence.

Purpose:

> Measure how much of the problem can be solved without cross-account reasoning.

### Baseline B — Graph structural-only

Uses structural connectivity information without the full behavioral feature set.

Purpose:

> Demonstrate the false-positive failure mode of raw structural reasoning on legitimate shared infrastructure.

### Baseline C — Full multi-signal

Uses the deployed multi-signal architecture:

- individual behavior;
- lifecycle timing;
- structural context;
- behavioral coordination evidence;
- deterministic rules / confirmation logic.

Purpose:

> Test whether the combined approach retains strong detection while controlling structural false positives.

---

## 6. Operating thresholds

Thresholds were selected on validation data only.

| Model | Operating threshold |
|---|---:|
| Baseline A | 0.50 |
| Baseline B | 2 structural members / cluster threshold |
| Baseline C | 0.50 |

For the binary ML models, a probability at or above `0.5` is treated as positive.

The threshold is a benchmark operating point, not a production calibration recommendation.

---

## 7. Final held-out results

The current persisted results are:

| Model | Precision | Recall | F1 | Accuracy | FP | FN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Individual-only | **100.00%** | 87.18% | 93.15% | 93.06% | 0 | 5 | 0.00% | 12.82% |
| Graph structural-only | 80.95% | 43.59% | 56.67% | 63.89% | **4** | **22** | **12.12%** | 56.41% |
| **Full multi-signal** | **100.00%** | **97.44%** | **98.70%** | **98.61%** | **0** | **1** | **0.00%** | **2.56%** |

### Confusion matrices

#### Individual-only

```text
TP = 34
TN = 33
FP = 0
FN = 5
```

#### Graph structural-only

```text
TP = 17
TN = 29
FP = 4
FN = 22
```

#### Full multi-signal

```text
TP = 38
TN = 33
FP = 0
FN = 1
```

---

## 8. Financial exposure results

The held-out test contains:

```text
Total abuse exposure = ₹10,378.11
```

### Abuse exposure captured by positive predictions

| Model | Captured exposure | Capture rate | Missed exposure |
|---|---:|---:|---:|
| Individual-only | ₹9,680.37 | 93.28% | ₹697.74 |
| Graph structural-only | ₹3,117.95 | 30.04% | ₹7,260.16 |
| **Full multi-signal** | **₹10,256.54** | **98.83%** | **₹121.57** |

This metric means:

> Of the labeled abuse refund amount in the held-out benchmark, how much belongs to cases that the classifier marked positive?

It does **not** mean that the corresponding amount was literally prevented or recovered.

### Important distinction

```text
Captured abuse exposure
        ≠
Guaranteed prevented money
        ≠
Recovered money
```

Operational prevention depends on processing latency, merchant response and downstream action, none of which are simulated by this benchmark metric.

---

## 9. False-positive cost sensitivity

The project deliberately keeps false-positive economics configurable rather than claiming a real merchant friction cost.

### Assumed ₹50 per false positive

| Model | FP count | Operational FP cost |
|---|---:|---:|
| Individual-only | 0 | ₹0 |
| Graph-only | 4 | ₹200 |
| Full multi-signal | 0 | ₹0 |

### Assumed ₹200 per false positive

| Model | FP count | Operational FP cost |
|---|---:|---:|
| Individual-only | 0 | ₹0 |
| Graph-only | 4 | ₹800 |
| Full multi-signal | 0 | ₹0 |

### Assumed ₹500 per false positive

| Model | FP count | Operational FP cost |
|---|---:|---:|
| Individual-only | 0 | ₹0 |
| Graph-only | 4 | ₹2,000 |
| Full multi-signal | 0 | ₹0 |

These are **sensitivity-analysis assumptions**, not observed merchant economics.

---

## 10. Per-scenario results

| Scenario | Baseline A | Baseline B | Baseline C |
|---|---:|---:|---:|
| `AS03_SHARED_PAYMENT_DEVICE_RING` | 72.22% recall | **94.44% recall** | **94.44% recall** |
| `AS04_ISOLATED_REFUND_CHURN` | **100.00% recall** | 0.00% recall | **100.00% recall** |
| `BACKGROUND` | 0.00% FPR | 0.00% FPR | 0.00% FPR |
| `LL02_FREQUENT_SHOPPER` | 0.00% FPR | 0.00% FPR | 0.00% FPR |
| `LL03_SHARED_HOUSEHOLD` | 0.00% FPR | **66.67% FPR** | **0.00% FPR** |

### What these scenarios test

**AS03** tests an unseen shared-payment/device coordination pattern.

**AS04** demonstrates that a purely structural detector can miss isolated behavioral abuse, while the individual model can still catch it.

**LL02** tests legitimate high activity.

**LL03** is the key structural lookalike: customers legitimately share household infrastructure. Graph-only produces false positives; the full system suppresses them.

---

## 11. Interpretation

The current benchmark supports three useful conclusions.

### Conclusion 1 — individual behavior matters

Baseline A reaches 87.18% recall on the held-out benchmark. It is already useful, but it misses part of the coordinated AS03 pattern.

### Conclusion 2 — structure alone is unsafe

Baseline B reaches only 43.59% recall and produces four false positives overall. On the shared-household lookalike it reaches a 66.67% FPR.

### Conclusion 3 — the combined architecture is stronger on this benchmark

Baseline C reaches 97.44% recall while producing zero false positives on the final held-out set and captures 98.83% of labeled abuse exposure.

The strongest architecture demonstration is therefore:

```text
individual-only
    ↓
strong but misses coordination

structural-only
    ↓
finds connections but overflags legitimate structure

full multi-signal
    ↓
uses behavior to confirm when structure should matter
```

---

## 12. What the benchmark supports

It supports the narrow claims that, on this controlled benchmark:

- the full model detects more labeled abuse cases than the individual-only baseline;
- graph-only reasoning creates a clear false-positive failure mode on the shared-household lookalike;
- behavioral confirmation eliminates that demonstrated LL03 false-positive pattern;
- the full system captures 98.83% of labeled abuse refund exposure in the held-out set at the selected operating point.

---

## 13. What the benchmark does not support

Do not use these results to claim:

- production Razorpay fraud accuracy;
- calibrated real-world fraud probability;
- universal evasion resistance;
- guaranteed prevention of captured exposure;
- automatic recovery of realized refunds;
- generalization to every merchant policy or product category.

The benchmark is synthetic and controlled.

---

## 14. Metric coverage and caveats

The architecture originally identified several additional metrics as desirable. The current persisted evaluator does **not** store all of them.

### Persisted now

- precision;
- recall;
- F1;
- accuracy;
- TP/TN/FP/FN;
- FPR/FNR;
- abuse exposure captured;
- missed abuse exposure;
- exposure capture rate;
- false-positive cost sensitivity;
- per-scenario recall/FPR.

### Not currently persisted

- PR-AUC;
- full precision-recall curves;
- top-K pending exposure curves by analyst capacity;
- alpha sensitivity curves;
- repeated-run confidence intervals;
- formal probability calibration metrics.

This document intentionally leaves those fields unreported rather than inventing values.

### Why this matters

The hackathon asks for honest metrics. A metric that was designed on paper but not actually computed should not appear as though it was measured.

---

## 15. Reproduction

From the repository root:

```powershell
python scripts\generate_datasets.py
python scripts\run_evaluation.py
```

The evaluator regenerates the training/evaluation partitions and writes:

```text
data/results.json
```

The console should show the three-model comparison.

The saved model artifact is regenerated under:

```text
backend/app/models/risk_model.json
```

---

## 16. Reporting checklist

Before submitting or presenting the benchmark:

- [ ] Confirm `data/results.json` is the file being presented.
- [ ] Re-run leakage checks if the simulator changes.
- [ ] Confirm thresholds were selected on validation only.
- [ ] Do not add production-sounding language to synthetic results.
- [ ] Keep the legitimate lookalike FPR visible.
- [ ] Keep the “captured exposure” vs “prevented money” distinction explicit.
- [ ] Do not report PR-AUC or top-K curves unless the evaluator has actually computed them.

---

## Final benchmark headline

> **On the current controlled held-out benchmark, Refund Sentinel's full multi-signal system achieved 97.44% recall, 100.00% precision, and 98.83% abuse-exposure capture, while the graph-only baseline produced a 66.67% false-positive rate on the legitimate shared-household lookalike.**

That is the project's defensible benchmark story.
