"""Train the Refund Sentinel analytical risk model.

The training set is generated from the project's deterministic simulator. Labels
are simulator ground truth used only for supervised training/evaluation; they
never enter the production feature vector.

Validation is group-aware: all examples from the same abuse ring, family, or
background customer remain entirely in either training or validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.app.finance.state_engine import FinancialStateEngine
from backend.app.ml.dataset import create_dataset
from backend.app.ml.features import build_feature_vector
from backend.app.ml.model import TrainingConfig
from backend.app.ml.persistence import PersistedModelBundle, save_model_bundle
from backend.app.ml.training import train_and_validate_model
from backend.app.risk.assessment import RiskAssessor
from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.labels import LabelClassification, SimulationOutput
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    AS02_VELOCITY_REFUND_ABUSE,
    AS03_SHARED_PAYMENT_DEVICE_RING,
    LL01_LEGITIMATE_FAMILY,
    LL02_FREQUENT_SHOPPER,
)


def _add_output(
    *,
    output: SimulationOutput,
    group_prefix: str,
    group_by_customer: bool,
    events: list,
    labels: list,
    refund_groups: dict[str, str],
) -> None:
    events.extend(output.events)
    labels.extend(output.labels)
    for label in output.labels:
        if label.refund_id is None:
            continue
        if group_by_customer:
            if label.customer_id is None:
                raise ValueError("Background refund label is missing customer_id")
            group = f"{group_prefix}:customer:{label.customer_id}"
        else:
            group = group_prefix
        refund_groups[str(label.refund_id)] = group


def _label_map(labels: list) -> dict[str, int]:
    result: dict[str, int] = {}
    for label in labels:
        if label.refund_id is None:
            continue
        result[str(label.refund_id)] = (
            1 if label.classification == LabelClassification.ABUSE else 0
        )
    return result


def _build_dataset(snapshot, labels: list, refund_groups: dict[str, str]):
    assessor = RiskAssessor(snapshot)
    feature_names: tuple[str, ...] | None = None
    rows: list[list[float]] = []
    targets: list[int] = []
    groups: list[str] = []
    skipped = 0
    label_map = _label_map(labels)

    for refund_id, target in label_map.items():
        matching = next((rid for rid in snapshot.refunds if str(rid) == refund_id), None)
        if matching is None or refund_id not in refund_groups:
            skipped += 1
            continue
        try:
            vector = build_feature_vector(assessor.assess(matching))
        except (ValueError, KeyError):
            skipped += 1
            continue
        if feature_names is None:
            feature_names = vector.feature_names
        elif vector.feature_names != feature_names:
            skipped += 1
            continue
        rows.append(vector.values)
        targets.append(target)
        groups.append(refund_groups[refund_id])

    if not rows:
        raise RuntimeError("No labelled refund feature vectors were produced")

    return feature_names or (), rows, targets, groups, skipped


def _persist_metrics(path: Path, result, *, dataset_rows: int, group_count: int, class_counts: Counter) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.validation_metrics
    payload["training_metadata"] = {
        "dataset_row_count": dataset_rows,
        "training_row_count": result.training_row_count,
        "validation_row_count": result.validation_row_count,
        "feature_count": len(result.model.feature_names),
        "group_count": group_count,
        "class_counts": {
            "abuse": int(class_counts[1]),
            "legitimate": int(class_counts[0]),
        },
        "split_strategy": "group_stratified",
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
        },
        "confusion_matrix": {
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    events: list = []
    labels: list = []
    refund_groups: dict[str, str] = {}

    print("\nRefund Sentinel — larger, group-aware ML training")
    print("=" * 58)

    background = BackgroundPopulationGenerator(seed=args.seed).generate(
        num_customers=args.customers,
        num_orders_per_customer=args.orders_per_customer,
        refund_probability=args.background_refund_probability,
    )
    _add_output(
        output=background,
        group_prefix="background",
        group_by_customer=True,
        events=events,
        labels=labels,
        refund_groups=refund_groups,
    )
    print(f"Background events: {len(background.events)}")

    scenario_specs = [
        ("AS01", AS01_DENSE_COORDINATED_REFUND_RING, args.as01_groups, {"num_customers": 5, "orders_per_customer": 3}),
        ("AS02", AS02_VELOCITY_REFUND_ABUSE, args.as02_groups, {"num_customers": 4, "refunds_per_customer": 6}),
        ("AS03", AS03_SHARED_PAYMENT_DEVICE_RING, args.as03_groups, {"num_customers": 6, "orders_per_customer": 3}),
        ("LL01", LL01_LEGITIMATE_FAMILY, args.ll01_groups, {"num_family_members": 4, "orders_per_member": 4}),
        ("LL02", LL02_FREQUENT_SHOPPER, args.ll02_groups, {"num_customers": 5, "orders_per_customer": 12}),
    ]

    for scenario_name, scenario_class, count, kwargs in scenario_specs:
        for index in range(count):
            output = scenario_class(seed=args.seed + 10_000 + len(events) + index * 97).generate(**kwargs)
            _add_output(
                output=output,
                group_prefix=f"{scenario_name}:{index}",
                group_by_customer=False,
                events=events,
                labels=labels,
                refund_groups=refund_groups,
            )
        print(f"{scenario_name} groups: {count}")

    print(f"\nReconstructing {len(events):,} events...")
    snapshot = FinancialStateEngine().reconstruct_from(events)
    print(f"Snapshot refunds: {len(snapshot.refunds):,}; anomalies: {len(snapshot.anomalies):,}")

    print("Building risk feature vectors...")
    feature_names, rows, targets, groups, skipped = _build_dataset(
        snapshot, labels, refund_groups
    )
    class_counts = Counter(targets)
    print(
        f"Dataset: {len(rows):,} examples × {len(feature_names)} features | "
        f"abuse={class_counts[1]:,}, legitimate={class_counts[0]:,} | "
        f"groups={len(set(groups)):,}"
    )
    if skipped:
        print(f"Skipped {skipped} labelled refunds that could not be assessed")

    dataset = create_dataset(
        feature_names=feature_names,
        feature_rows=rows,
        labels=targets,
    )

    result = train_and_validate_model(
        dataset,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        groups=groups,
        training_config=TrainingConfig(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2_regularization=args.l2,
        ),
    )

    metrics = result.validation_metrics
    print("\nHeld-out group-aware validation")
    print(f"Training rows:   {result.training_row_count:,}")
    print(f"Validation rows: {result.validation_row_count:,}")
    print(f"Accuracy:        {metrics.accuracy:.1%}")
    print(f"Precision:       {metrics.precision:.1%}")
    print(f"Recall:          {metrics.recall:.1%}")
    print(
        "Confusion matrix: "
        f"TP={metrics.true_positives} TN={metrics.true_negatives} "
        f"FP={metrics.false_positives} FN={metrics.false_negatives}"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = PersistedModelBundle.from_training_result(result)
    save_model_bundle(bundle, output)
    _persist_metrics(
        output,
        result,
        dataset_rows=len(rows),
        group_count=len(set(groups)),
        class_counts=class_counts,
    )
    print(f"\nModel artifact saved to: {output.resolve()}")
    print("Set ML_MODEL_PATH to this path and restart the backend.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="backend/app/models/risk_model.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--customers", type=int, default=1500)
    parser.add_argument("--orders-per-customer", type=int, default=4)
    parser.add_argument("--background-refund-probability", type=float, default=0.35)
    parser.add_argument("--as01-groups", type=int, default=20)
    parser.add_argument("--as02-groups", type=int, default=20)
    parser.add_argument("--as03-groups", type=int, default=20)
    parser.add_argument("--ll01-groups", type=int, default=20)
    parser.add_argument("--ll02-groups", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.customers = 120
        args.orders_per_customer = 3
        args.as01_groups = args.as02_groups = args.as03_groups = 3
        args.ll01_groups = args.ll02_groups = 3
        args.epochs = min(args.epochs, 500)
    return args


if __name__ == "__main__":
    train(parse_args())
