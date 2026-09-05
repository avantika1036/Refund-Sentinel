"""Dataset generator CLI script for Refund Sentinel.

Generates standard train, eval, and held-out test datasets using the
simulator framework and saves them to disk as JSONL files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.ml.features import build_feature_vector
from backend.app.risk.assessment import RiskAssessor
from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.labels import LabelClassification, SimulationOutput
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    AS02_VELOCITY_REFUND_ABUSE,
    AS03_SHARED_PAYMENT_DEVICE_RING,
    AS04_ISOLATED_REFUND_CHURN,
    LL01_LEGITIMATE_FAMILY,
    LL02_FREQUENT_SHOPPER,
    LL03_SHARED_HOUSEHOLD,
)


def generate_partition(
    num_background_customers: int,
    include_held_out: bool = False,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Generate simulation events, state snapshot, and labeled feature rows."""
    events = []
    labels = []

    # 1. Background legitimate baseline
    bg_gen = BackgroundPopulationGenerator(seed=seed)
    bg_out = bg_gen.generate(num_customers=num_background_customers)
    events.extend(bg_out.events)
    labels.extend(bg_out.labels)

    # 2. Scenarios
    # Training/validation use the known scenario families. The held-out test
    # partition deliberately uses only unseen scenario families, preventing the
    # evaluation from reporting scenario memorization as generalization.
    if include_held_out:
        scenarios = [
            AS03_SHARED_PAYMENT_DEVICE_RING,
            AS04_ISOLATED_REFUND_CHURN,
            LL02_FREQUENT_SHOPPER,
            LL03_SHARED_HOUSEHOLD,
        ]
    else:
        scenarios = [
            AS01_DENSE_COORDINATED_REFUND_RING,
            AS02_VELOCITY_REFUND_ABUSE,
            LL01_LEGITIMATE_FAMILY,
        ]

    # Each generator receives a distinct deterministic seed. Reusing one seed
    # would recreate the same deterministic EventIds across scenario families.
    for scenario_index, scenario_cls in enumerate(scenarios, start=1):
        scenario = scenario_cls(seed=seed + scenario_index * 10_000)
        sc_out = scenario.generate()
        events.extend(sc_out.events)
        labels.extend(sc_out.labels)

    # Sort events chronologically
    events.sort(key=lambda e: e.envelope.occurred_at.value)

    # Reconstruct state
    engine = FinancialStateEngine()
    snapshot = engine.reconstruct_from(events)

    # Label map
    label_map = {
        str(l.refund_id): (1 if l.classification == LabelClassification.ABUSE else 0)
        for l in labels if l.refund_id is not None
    }
    scenario_map = {
        str(l.refund_id): (l.scenario_type.value if hasattr(l.scenario_type, "value") else str(l.scenario_type))
        for l in labels if l.refund_id is not None
    }

    # Extract features
    assessor = RiskAssessor(snapshot)
    dataset_rows = []
    feature_names = None

    for refund_id, refund in snapshot.refunds.items():
        rid_str = str(refund_id)
        if rid_str not in label_map:
            continue

        assessment = assessor.assess(refund_id)
        fvec = build_feature_vector(assessment)
        if feature_names is None:
            feature_names = list(fvec.feature_names)

        dataset_rows.append({
            "refund_id": rid_str,
            "customer_id": str(refund.customer_id),
            "requested_amount_paise": refund.requested_amount.amount_paise,
            "scenario": scenario_map.get(rid_str, "background"),
            "label": label_map[rid_str],
            "feature_names": feature_names,
            "feature_values": fvec.values,
            "triggered_rule_count": sum(
                1 for output in assessment.rule_outputs if output.triggered
            ),
        })

    summary = {
        "total_events": len(events),
        "total_refunds": len(dataset_rows),
        "abuse_count": sum(r["label"] for r in dataset_rows),
        "legit_count": sum(1 for r in dataset_rows if r["label"] == 0),
    }

    return dataset_rows, summary


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic train/eval datasets.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--train-customers", type=int, default=100)
    parser.add_argument("--eval-customers", type=int, default=40)
    parser.add_argument("--test-customers", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating training dataset...")
    train_rows, train_sum = generate_partition(args.train_customers, include_held_out=False, seed=42)
    with open(out_dir / "train.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Train dataset created: {train_sum}")

    print("Generating validation dataset...")
    eval_rows, eval_sum = generate_partition(args.eval_customers, include_held_out=False, seed=101)
    with open(out_dir / "eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Eval dataset created: {eval_sum}")

    print("Generating held-out test dataset...")
    test_rows, test_sum = generate_partition(args.test_customers, include_held_out=True, seed=202)
    with open(out_dir / "test_heldout.jsonl", "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Test held-out dataset created: {test_sum}")


if __name__ == "__main__":
    main()
