# Refund Sentinel — Architectural Decisions

> This document records the decisions that define the current project contract, the reason each decision exists, and the trade-offs that were accepted.

## Table of Contents

- [ADR-001 — Scope the system to coordinated refund abuse](#adr-001--scope-the-system-to-coordinated-refund-abuse)
- [ADR-002 — Make refund lifecycle state a first-class signal](#adr-002--make-refund-lifecycle-state-a-first-class-signal)
- [ADR-003 — Use an append-oriented event ledger](#adr-003--use-an-append-oriented-event-ledger)
- [ADR-004 — Keep financial truth deterministic](#adr-004--keep-financial-truth-deterministic)
- [ADR-005 — Separate structural graph from behavioral evidence](#adr-005--separate-structural-graph-from-behavioral-evidence)
- [ADR-006 — Require behavioral confirmation before graph amplification](#adr-006--require-behavioral-confirmation-before-graph-amplification)
- [ADR-007 — Rules produce evidence signals, not probabilities](#adr-007--rules-produce-evidence-signals-not-probabilities)
- [ADR-008 — Keep financial exposure outside the ML feature vector](#adr-008--keep-financial-exposure-outside-the-ml-feature-vector)
- [ADR-009 — Prefer the simplest model that proves the claim](#adr-009--prefer-the-simplest-model-that-proves-the-claim)
- [ADR-010 — Use a customer/cluster-disjoint held-out benchmark](#adr-010--use-a-customercluster-disjoint-held-out-benchmark)
- [ADR-011 — Select thresholds on validation only](#adr-011--select-thresholds-on-validation-only)
- [ADR-012 — Include legitimate structural lookalikes](#adr-012--include-legitimate-structural-lookalikes)
- [ADR-013 — Make the LLM a narrative layer only](#adr-013--make-the-llm-a-narrative-layer-only)
- [ADR-014 — Never permanently cache provider fallbacks](#adr-014--never-permanently-cache-provider-fallbacks)
- [ADR-015 — Preserve mitigating evidence](#adr-015--preserve-mitigating-evidence)
- [ADR-016 — Keep synthetic-data claims narrow](#adr-016--keep-synthetic-data-claims-narrow)
- [ADR-017 — Report actual metrics only](#adr-017--report-actual-metrics-only)
- [ADR-018 — Use a React operator console while preserving backend authority](#adr-018--use-a-react-operator-console-while-preserving-backend-authority)
- [ADR-019 — Keep the benchmark and demo data conceptually separate](#adr-019--keep-the-benchmark-and-demo-data-conceptually-separate)
- [ADR-020 — Defer GNN / sequence complexity until evidence justifies it](#adr-020--defer-gnn--sequence-complexity-until-evidence-justifies-it)

---

## ADR-001 — Scope the system to coordinated refund abuse

**Status:** Accepted

**Decision:** Build one detector for coordinated post-payment refund abuse rather than a generic fraud platform.

**Why:** Track 02 rewards solving a concrete loss class with measurable precision/recall and financial impact. A narrow detector makes the benchmark falsifiable and the demo understandable.

**Consequence:** Generic payment fraud, checkout authorization and broad abuse domains remain outside the current product boundary.

---

## ADR-002 — Make refund lifecycle state a first-class signal

**Status:** Accepted

**Decision:** Treat payment capture, order state, delivery state, refund request and refund processing as a lifecycle rather than independent rows.

**Why:** A refund request's meaning depends on where it occurs in the payment/order/refund sequence. Cross-account alignment in lifecycle timing is a core differentiator from generic graph analysis.

**Consequence:** Feature computation must preserve timestamps and point-in-time semantics.

---

## ADR-003 — Use an append-oriented event ledger

**Status:** Accepted

**Decision:** Persist domain events and reconstruct current financial state from the event history.

**Why:** Event-driven reasoning supports idempotency, replay handling, out-of-order events and auditability.

**Consequence:** Current state is derived deterministically instead of trusting one mutable summary row.

---

## ADR-004 — Keep financial truth deterministic

**Status:** Accepted

**Decision:** Payment, capture, refund and remaining-refundable calculations are computed by deterministic domain logic, not ML or an LLM.

**Why:** Financial correctness is an invariant, not a prediction problem.

**Consequence:** An unavailable model or LLM can degrade risk/explanation quality without changing the underlying money calculation.

---

## ADR-005 — Separate structural graph from behavioral evidence

**Status:** Accepted

**Decision:** The structural graph contains facts about relationships; coordination evidence is calculated separately from lifecycle and behavioral data.

**Why:** Embedding suspicious behavior directly into the graph makes the representation partly circular. The graph should say who is connected, not who is suspicious.

**Consequence:** Graph structure can be inspected independently and used as a model feature rather than a verdict.

---

## ADR-006 — Require behavioral confirmation before graph amplification

**Status:** Accepted

**Decision:** Structural cluster features contribute only when behavioral coordination evidence passes a configurable confirmation threshold.

**Why:** This is the architectural defense against false positives from legitimate families, offices and shared infrastructure.

**Consequence:** A large graph component with heterogeneous behavior should not automatically become high risk.

---

## ADR-007 — Rules produce evidence signals, not probabilities

**Status:** Accepted

**Decision:** Rules expose deterministic triggers, observed values, thresholds and signal weights.

**Why:** “The rule fired” is deterministic. “The customer is fraudulent with 100% confidence” is not.

**Consequence:** Rule output remains explainable and is not mislabeled as calibrated fraud probability.

---

## ADR-008 — Keep financial exposure outside the ML feature vector

**Status:** Accepted

**Decision:** Risk is estimated first; exposure is applied afterward.

**Why:** Including raw money values in the model risks conflating “high value” with “high risk” and makes the risk model harder to interpret.

**Consequence:** Financial exposure can be adjusted as a policy layer without retraining the risk model.

---

## ADR-009 — Prefer the simplest model that proves the claim

**Status:** Accepted

**Decision:** Use the custom logistic regression implementation as the deployed analytical model.

**Why:** It is reproducible, interpretable, fast and sufficient for the current benchmark. More complex models are not justified by complexity alone.

**Consequence:** The repository does not require a heavyweight ML runtime for inference.

---

## ADR-010 — Use a customer/cluster-disjoint held-out benchmark

**Status:** Accepted

**Decision:** Prevent customers and clusters from crossing train/validation/test boundaries.

**Why:** Row-level random splits can leak the same entity or ring into multiple partitions and inflate metrics.

**Consequence:** The split is stricter and more representative of unseen-entity evaluation.

---

## ADR-011 — Select thresholds on validation only

**Status:** Accepted

**Decision:** Operating thresholds are selected using validation data and then frozen for held-out reporting.

**Why:** Using the held-out test set to tune the threshold contaminates the final benchmark.

**Consequence:** Held-out labels are used only for final measurement.

---

## ADR-012 — Include legitimate structural lookalikes

**Status:** Accepted

**Decision:** The held-out benchmark includes scenarios such as shared households and legitimate high activity.

**Why:** A detector that only sees clean negatives can appear accurate while failing the real business problem.

**Consequence:** False-positive behavior becomes a visible part of the evaluation.

---

## ADR-013 — Make the LLM a narrative layer only

**Status:** Accepted

**Decision:** The LLM receives a structured evidence bundle and returns a narrative explanation. It does not mutate financial state, override risk or call financial APIs.

**Why:** This gives the product an AI investigation interface without making probabilistic text generation a critical dependency for financial correctness.

**Consequence:** LLM failures degrade explanation quality but should not break the risk workflow.

---

## ADR-014 — Never permanently cache provider fallbacks

**Status:** Accepted

**Decision:** Only successful provider-generated explanations may be cached. Rate-limit or error fallbacks are never permanently cached for the evidence lifetime.

**Why:** A transient 404, 429 or network error must not prevent a later request from retrying after the provider/key becomes available.

**Consequence:** Repeated refreshes after a transient provider outage remain retryable.

---

## ADR-015 — Preserve mitigating evidence

**Status:** Accepted

**Decision:** Evidence bundles include both suspicious and mitigating signals.

**Why:** Investigators need to understand not only why a case was escalated, but also what evidence argues against escalation.

**Consequence:** Case explanations become more auditable and less one-sided.

---

## ADR-016 — Keep synthetic-data claims narrow

**Status:** Accepted

**Decision:** Synthetic benchmark results are described as controlled proof-of-concept measurements.

**Why:** Production fraud behavior and labels are unavailable in the prototype.

**Consequence:** Documentation explicitly avoids claiming production accuracy.

---

## ADR-017 — Report actual metrics only

**Status:** Accepted

**Decision:** If the current evaluator does not compute a metric, the documentation does not report a fabricated value.

**Why:** Honest measurement is itself part of the hackathon requirement.

**Consequence:** PR-AUC, full top-K pending-exposure curves and calibration metrics remain explicitly marked as not persisted rather than being invented.

---

## ADR-018 — Use a React operator console while preserving backend authority

**Status:** Accepted

**Decision:** Use React + TypeScript for the judge-facing interface while keeping financial and risk computation in the backend.

**Why:** A polished operator experience materially improves demonstrability, but UI code should not become the source of truth.

**Consequence:** The frontend consumes API outputs and presents them; it does not independently calculate authoritative financial state.

---

## ADR-019 — Keep the benchmark and demo data conceptually separate

**Status:** Accepted

**Decision:** The benchmark used to measure model quality is separate from the synthetic data seeded into the live investigation queue.

**Why:** A benchmark should remain an evaluation artifact; the demo database is a product-experience artifact.

**Consequence:** The queue can demonstrate the workflow without being mistaken for the held-out test set.

---

## ADR-020 — Defer GNN / sequence complexity until evidence justifies it

**Status:** Accepted

**Decision:** Do not add GNNs or sequence models unless the current architecture leaves a meaningful measured performance gap.

**Why:** Complexity without measured benefit makes a hackathon system harder to explain and easier to overfit.

**Consequence:** Any future advanced model must beat the current baseline on a valid held-out experiment before it replaces the simpler implementation.

---

# Decision summary

The current design can be summarized as:

```text
Real events
   ↓
Deterministic financial state
   ↓
Lifecycle + behavior
   ↓
Structural relationships
   ↓
Behavioral confirmation
   ↓
Risk signal
   ↓
Financial exposure
   ↓
Evidence bundle
   ↓
LLM narrative
   ↓
Human investigation
```

The governing principle is:

> **Use AI to improve risk prioritization and investigation, but never allow an opaque AI component to become the authority for financial truth.**
