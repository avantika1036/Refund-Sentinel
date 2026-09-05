# Refund Sentinel — Architecture, Threat Model, Data Card & Model Card

> **Track 02 — AI Risk Manager**  
> A refund-lifecycle-aware coordinated refund-abuse detector and investigation system.

## Table of Contents

- [System goals](#system-goals)
- [Non-goals and boundaries](#non-goals-and-boundaries)
- [Architecture overview](#architecture-overview)
- [Repository architecture](#repository-architecture)
- [Event and financial model](#event-and-financial-model)
- [Refund lifecycle reasoning](#refund-lifecycle-reasoning)
- [Structural graph](#structural-graph)
- [Behavioral confirmation](#behavioral-confirmation)
- [Deterministic rules](#deterministic-rules)
- [Feature architecture](#feature-architecture)
- [Machine-learning model card](#machine-learning-model-card)
- [Risk-score composition](#risk-score-composition)
- [Financial exposure architecture](#financial-exposure-architecture)
- [Evidence bundle](#evidence-bundle)
- [LLM boundary](#llm-boundary)
- [API and frontend architecture](#api-and-frontend-architecture)
- [Threat model](#threat-model)
- [Reliability and failure handling](#reliability-and-failure-handling)
- [Privacy and fairness](#privacy-and-fairness)
- [Data card](#data-card)
- [Reproducibility and auditability](#reproducibility-and-auditability)
- [Technology stack](#technology-stack)
- [Production evolution](#production-evolution)



---

## 1. System goals

Refund Sentinel exists to answer four operational questions:

1. Is this refund individually risky?
2. Is this refund part of a broader coordinated pattern?
3. What financial exposure is associated with the pattern?
4. Which cases deserve investigation first?

The system is designed around the observation that coordinated refund abuse can be invisible at the individual-transaction level.

### Design principles

- **Financial correctness is deterministic.**
- **Risk is probabilistic / signal-based, not proof of fraud.**
- **Structural relationships are not inherently suspicious.**
- **Behavior must confirm structure before cluster evidence is amplified.**
- **The LLM is never authoritative for money or risk.**
- **All benchmark claims come from controlled held-out data.**
- **Failures and uncertainty remain visible.**

---

## 2. Non-goals and boundaries

Refund Sentinel is not:

- a generic payment fraud classifier;
- a checkout authorization engine;
- a customer accusation system;
- a graph visualization with arbitrary risk labels;
- a chatbot placed on top of a refund API;
- an automated refund blocker;
- a production-ready multi-tenant risk platform;
- a claim that synthetic benchmark performance equals production fraud performance.

The prototype focuses on **coordinated post-payment refund abuse**.

---

## 3. Architecture overview

<img width="612" height="642" alt="image" src="https://github.com/user-attachments/assets/60af25a2-4577-4790-8ddb-58e154a635cf" />


---

### Architectural authority order

```text
Source events
    ↓
Financial reconstruction
    ↓
Deterministic evidence + model inference
    ↓
Operational priority
    ↓
Evidence bundle
    ↓
Human-readable LLM narrative
```

The LLM sits at the end of the chain.

---

## 4. Repository architecture

```text
backend/app/
├── api/             # HTTP routes and request/response schemas
├── domain/          # entities, enums, events, identifiers, value objects
├── finance/         # state reconstruction, financial invariants, exposure
├── graph/           # structural graph and connected-component extraction
├── investigator/    # evidence bundle + LLM/fallback explanation
├── ml/              # dataset, preprocessing, logistic model, inference
├── persistence/     # SQLAlchemy models, database and ingestion services
├── risk/            # features, rules, confirmation, scoring, investigations
├── simulator/       # reproducible background + abuse/lookalike scenarios
├── config.py
├── main.py
└── models/risk_model.json
```

The frontend is a React + TypeScript application under `frontend/src/`.

---

## 5. Event and financial model

### Event lifecycle

The project treats payment/refund activity as domain events rather than a single mutable refund row.

```text
payment.created
       ↓
payment.captured
       ↓
order lifecycle
       ↓
refund.requested
       ↓
refund.created
       ↓
refund.processed
```

The financial state engine reconstructs state from ordered events and preserves deterministic accounting rules.

### Money representation

Financial quantities are handled as integer paise in domain calculations where monetary precision matters.

This avoids binary floating-point ambiguity in financial invariants.

### Example invariant

```text
Captured:  ₹10,000
Refunded:  ₹7,000
Requested: ₹5,000

Remaining refundable = ₹3,000
```

The ML layer is never used to infer this value.

---

## 6. Refund lifecycle reasoning

Lifecycle features are the project's primary differentiator over generic graph detection.

### Important lifecycle signals

| Feature | Meaning |
|---|---|
| `capture_to_refund_latency_hrs` | Time between capture and refund request |
| `order_to_refund_latency_hrs` | Time between order creation and refund request |
| `delivery_to_refund_latency_hrs` | Time between delivery and refund request when delivery exists |
| `refund_amount_fraction` | Refund amount divided by captured amount |
| `is_full_refund` | Full-refund indicator |
| lifecycle timing alignment | Whether different accounts behave similarly in the same lifecycle |

The key distinction is not simply “fast refund = bad.” The useful signal is whether timing patterns become unusually coordinated across multiple accounts.

### Point-in-time rule

Features for an event are computed from information available **at or before the scoring timestamp**.

This prevents later events from leaking future knowledge into earlier decisions.

---

## 7. Structural graph

The graph represents relationships as structural facts, not suspicion.

### Current structural entities

The implementation materially uses:

- Customer;
- Device;
- Address;
- Payment;
- Order;
- Refund.

The core customer-to-identity relationships include:

```text
Customer --USES--> Device
Order --SHIPS_TO--> Address
Customer --OWNS--> Payment
Payment --BELONGS_TO--> Order
Refund --TARGETS--> Payment
```

### Connected components

Components are extracted deterministically through adjacency traversal / DFS.

A component answers:

> Which customers are structurally connected through supported relationships?

It does **not** answer:

> Are these customers fraudulent?

### Why raw structural graph is insufficient

A family can share a device and address.

An office can share infrastructure.

A coordinated abuse ring can also share infrastructure.

The structural graph alone cannot distinguish them reliably. That distinction is delegated to the behavioral confirmation layer.

---

## 8. Behavioral confirmation

Behavioral confirmation is the architectural gate that prevents structure from dominating the score without corroboration.

### Inputs

The current confirmation layer uses signals including:

- lifecycle timing alignment;
- temporal burst concentration;
- refund-reason similarity;
- active-refund concentration / neighborhood context.

### Gate

Conceptually:

```text
behavioral_confirmation_score < threshold
        ↓
cluster features suppressed

behavioral_confirmation_score >= threshold
        ↓
cluster features contribute proportionally
```

This preserves an important semantic distinction:

```text
shared infrastructure ≠ abuse
```

### Demonstration case: legitimate family

```text
Shared device + address
        ↓
Structural cluster exists
        ↓
Lifecycle timing varies
Reasons vary
No temporal burst
        ↓
Behavioral confirmation fails
        ↓
Cluster amplification suppressed
```

### Demonstration case: coordinated ring

```text
Shared relationships
        ↓
Tight lifecycle alignment
Strong temporal concentration
Consistent behavioral pattern
        ↓
Behavioral confirmation passes
        ↓
Cluster evidence contributes
```

---

## 9. Deterministic rules

Rules are interpretable evidence signals, not probabilities of fraud.

The current rule layer includes:

| Rule | Signal |
|---|---|
| R01 | Rapid refund after capture |
| R02 | Refund requested before delivery confirmation |
| R03 | Full refund on first refund for the payment |
| R04 | Elevated customer refund rate |
| R05 | Refund velocity spike |
| R06 | Multiple individually flagged accounts in the same cluster |

Each rule produces information such as:

```text
rule_id
triggered
evidence_type
evidence_value
evidence_threshold
base_signal_weight
notes
```

`base_signal_weight` is **signal strength**, not fraud probability.

---

## 10. Feature architecture

The runtime feature design separates signals by scope.

### Individual-level

Examples include:

- capture-to-refund latency;
- order-to-refund latency;
- delivery-to-refund latency and missingness;
- refund amount fraction;
- full-refund state;
- customer refund rate;
- customer refund velocity;
- reason code;
- reason rotation;
- account age;
- prior successful orders.

### Cluster-level

Examples include:

- structural cluster size;
- active-refund fraction;
- lifecycle timing alignment;
- temporal burst score;
- reason similarity;
- amount concentration.

### Relationship-level

Examples include:

- shared attribute type count;
- active-refund neighborhood count.

### Financial-level

Examples include:

- pending refund amount;
- realized refunded amount;
- remaining refundable amount.

Financial fields are deliberately **not ML inputs**. They are used after the model estimates risk so the system does not learn the trivial shortcut “larger amount = more fraud.”

---

## 11. Machine-learning model card

### Model type

**Binary logistic regression**, implemented in the repository without a heavyweight ML framework dependency.

### Input

The persisted model currently expects **46 preprocessed numeric inputs**.

### Output

A positive-class probability in `[0, 1]`.

The value is presented as an additional analytical signal, not as verified fraud probability.

### Training

Training is performed on the synthetic training partition with preprocessing fitted only on training data.

### Threshold

The current held-out evaluation selects an operating threshold from validation data. The current selected threshold is `0.5` for the full system.

### Strengths

- interpretable coefficients;
- fast inference;
- reproducible training;
- suitable for small/medium tabular data;
- easy to compare against ablations and baselines.

### Weaknesses

- linear decision boundary after feature transformation;
- calibration is not formally established;
- performance depends strongly on simulator design;
- no guarantee of production generalization;
- advanced graph or sequence representation learning is intentionally out of P0 scope.

### Why no GNN in P0?

The project does not use a GNN merely because the data contains a graph. The baseline comparison first tests whether structural and behavioral graph-derived features add measurable value. A more complex model should only be introduced if the benchmark demonstrates a meaningful gap.

---

## 12. Risk-score composition

The architecture intentionally exposes multiple signals rather than collapsing everything into a single opaque model output.

Conceptually:

```text
Rules
   +
Behavioral confirmation
   +
ML analytical probability
   ↓
Operational risk / priority layer
```

The important semantic separation is:

```text
ML probability ≠ final business decision
```

The final operational score remains accompanied by evidence and financial context.

---

## 13. Financial exposure architecture

### Realized suspicious refund amount

```text
sum(refunded_amount_i) × risk_score
```

Labeled in the UI as already processed suspicious exposure. It is not presented as guaranteed recoverable money.

### Pending refund exposure

```text
sum(requested_but_unprocessed_amount_i) × risk_score
```

This is the primary actionable financial context for an investigation queue.

### Remaining refundable exposure

```text
sum(remaining_refundable_amount_i) × risk_score
```

This is a forward-looking exposure estimate and is not a prediction that future fraud will occur.

---

## 14. Evidence bundle

The evidence bundle is the bridge between deterministic computation and the LLM.

It contains, at minimum:

```text
case / trigger information
refund lifecycle
individual behavioral evidence
cluster / relationship evidence
mitigating evidence
financial context
risk output
source-event references
```

### Why include mitigating evidence?

An investigation system is more credible when it records not only:

```text
What looks suspicious?
```

but also:

```text
What argues against escalation?
```

Examples include:

- varied refund reasons;
- high timing variance;
- established purchase history;
- product-defect context;
- seasonal context;
- missing data that limits confidence.

---

## 15. LLM boundary

The LLM receives a serialized evidence bundle and is asked to produce a structured investigation explanation.

### Allowed role

```text
Evidence bundle
      ↓
LLM summarization
      ↓
Human-readable narrative
```

### Forbidden role

```text
LLM
  X financial state
  X refund execution
  X risk override
  X policy override
  X evidence fabrication
```

### Provider failure

The current implementation falls back to deterministic narrative generation when:

- no API key is present;
- provider call fails;
- provider rate-limits the request;
- malformed output is returned;
- network errors occur.

Only successful provider-generated explanations are cached.

---

## 16. API and frontend architecture

### Main backend responsibilities

The FastAPI layer exposes endpoints for:

- health checks;
- investigation queue retrieval;
- individual investigation details;
- model evaluation results;
- webhook ingestion;
- assessment / risk operations.

### Frontend views

The current React application presents:

1. **Investigation Queue** — ranked cases and exposure context.
2. **Investigation Workspace** — risk, rules, evidence, graph relationships and explanation.
3. **Model Evaluation** — persisted held-out baseline comparison and exposure metrics.

The frontend uses live API data rather than hardcoded display values for the core views.

---

## 17. Threat model

### Threat: legitimate shared infrastructure

**Example:** family or office members share a device/address.

**Mitigation:** structural evidence requires behavioral confirmation before amplification.

### Threat: coordinated refund ring

**Example:** multiple accounts align lifecycle timing and refund behavior.

**Mitigation:** cross-account lifecycle and temporal features + structural neighborhood evidence.

### Threat: identifier evasion

**Example:** actors rotate devices or addresses.

**Mitigation:** do not rely on a single identifier; retain individual and temporal signals.

### Threat: reason-code evasion

**Example:** actors rotate refund reasons.

**Mitigation:** reason rotation is itself a modeled behavioral signal rather than a single static rule.

### Threat: prompt injection

**Example:** customer text attempts to instruct the LLM to approve a refund.

**Mitigation:** customer text is data, not privileged instructions; LLM input is structured evidence.

### Threat: malformed LLM output

**Example:** generated text references unsupported entities.

**Mitigation:** validate structured response and retain the evidence bundle as authoritative output.

### Threat: duplicate webhook delivery

**Example:** the same external event is delivered twice.

**Mitigation:** idempotent event ingestion and duplicate tracking.

### Threat: out-of-order events

**Example:** refund processing arrives before another lifecycle event.

**Mitigation:** financial state is reconstructed from event history rather than assuming arrival order is always semantic order.

### Threat: secrets exposure

**Example:** provider keys or webhook secrets committed to source control.

**Mitigation:** `.env` remains local; `.env.example` contains only placeholders.

---

## 18. Reliability and failure handling

| Failure | Expected behavior |
|---|---|
| ML artifact unavailable | Conservative fallback behavior; model signal unavailable |
| Graph data unavailable | Score continues using available individual/lifecycle evidence; confidence reduced |
| LLM unavailable | Structured evidence + heuristic narrative |
| Duplicate event | Mark duplicate; avoid double financial processing |
| Out-of-order event | Accept into ledger; reconstruct state deterministically |
| Missing optional identifier | Do not fabricate; reduce structural confidence |
| Provider 429 | Do not cache the fallback; later requests can retry |

The system is designed so the LLM is not a critical dependency for financial correctness.

---

## 19. Privacy and fairness

The prototype uses synthetic data. Production deployment would require:

- data minimization;
- strict access controls;
- tenant isolation;
- retention policy;
- secure secret management;
- audit controls.

### Fairness principle

Shared infrastructure can correlate with socioeconomic, geographic or household attributes. The system therefore avoids demographic or protected attributes and explicitly benchmarks legitimate shared-household scenarios.

The system should never interpret a shared address, device or network as sufficient proof of abuse.

---

## 20. Data card

### Dataset purpose

Synthetic supervised benchmark for coordinated refund-abuse detection and false-positive analysis.

### Unit of analysis

Refund/event-linked investigation rows with associated lifecycle, customer, structural and cluster features.

### Positive class

Simulator-labeled abuse scenarios.

### Negative class

Background plus legitimate lookalike scenarios.

### Construction

The simulator generates domain events, from which features are computed. Labels are maintained separately from feature computation.

### Split strategy

- customer-disjoint;
- cluster-disjoint;
- validation for threshold selection;
- held-out test for final metrics;
- held-out mechanisms / combinations / parameterizations to test generalization.

### Known risks

Because the benchmark is synthetic, the results reflect the simulator's assumptions as much as the model's ability.

---

## 21. Reproducibility and auditability

The system supports a traceable chain:

```text
priority score
    ← operational risk
    ← ML analytical signal + deterministic evidence
    ← named feature vector
    ← point-in-time event-derived features
    ← structural component + behavioral confirmation
    ← source events
```

Every risk decision should be interpreted together with:

- model version;
- feature version;
- policy / threshold context;
- evidence;
- source-event references.

The synthetic generator is seeded for deterministic reproduction of the benchmark and demo scenarios.

---

## 22. Technology stack

| Area | Technology |
|---|---|
| Runtime | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Validation/config | Pydantic + pydantic-settings |
| Persistence | SQLAlchemy + psycopg3 |
| Database | PostgreSQL 16; SQLite-compatible local configuration |
| Migrations | Alembic |
| Graph | Typed Python structures + deterministic DFS |
| ML | Custom logistic regression + preprocessing |
| Frontend | React 19 + TypeScript 6 |
| Build | Vite 8 |
| Testing | pytest + pytest-asyncio |
| AI narrative | Gemini / OpenAI-compatible HTTP integration + deterministic fallback |
| Infrastructure | Docker / Docker Compose |

---

## 23. Production evolution

A production architecture would add:

- authenticated multi-tenant access;
- merchant-specific model calibration;
- model monitoring and drift detection;
- richer identity graph coverage;
- high-degree graph controls;
- online feature computation;
- analyst outcome feedback;
- formal PR-AUC / calibration / top-K exposure studies;
- stronger secret management;
- durable LLM result storage and observability;
- data retention and deletion workflows.

The prototype intentionally stays narrower so the hackathon system remains explainable and measurable.
