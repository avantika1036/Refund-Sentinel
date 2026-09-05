"""Comparative evaluation runner for Refund Sentinel.

The benchmark uses three partitions:
- train: known scenario families for fitting,
- validation: the same known families with independent seeds for threshold selection,
- held-out: unseen abuse and legitimate families for final reporting.

The held-out suite intentionally contains both structural and non-structural abuse,
plus a legitimate shared-household lookalike. This makes it possible to measure
what graph topology alone misses or over-flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.baselines import (
    BaselineAIndividualOnly,
    BaselineBGraphHeuristicOnly,
    BaselineCFullSystem,
)
from backend.app.ml.dataset import MLDataset, create_dataset
from backend.app.ml.evaluation import EvaluationMetrics, compute_binary_metrics
from generate_datasets import generate_partition


def _dataset_from_rows(rows: list[dict], feature_names: list[str]) -> MLDataset:
    return create_dataset(
        feature_names=feature_names,
        feature_rows=[r["feature_values"] for r in rows],
        labels=[r["label"] for r in rows],
    )


def _select_probability_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> tuple[float, EvaluationMetrics]:
    """Select a validation threshold without looking at held-out labels.

    F1 is the primary objective. Precision is the tie-breaker, followed by the
    threshold closest to 0.5 so that an equivalent validation result does not
    choose an unnecessarily extreme operating point.
    """
    best_threshold = 0.5
    best_metrics: EvaluationMetrics | None = None
    for step in range(5, 100, 5):
        threshold = step / 100
        predictions = [1 if p >= threshold else 0 for p in probabilities]
        metrics = compute_binary_metrics(y_true=labels, y_pred=predictions)
        if best_metrics is None:
            best_threshold, best_metrics = threshold, metrics
            continue
        candidate_key = (
            metrics.f1_score,
            metrics.precision,
            metrics.recall,
            -abs(threshold - 0.5),
        )
        best_key = (
            best_metrics.f1_score,
            best_metrics.precision,
            best_metrics.recall,
            -abs(best_threshold - 0.5),
        )
        if candidate_key > best_key:
            best_threshold, best_metrics = threshold, metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _select_structural_threshold(
    dataset: MLDataset,
) -> tuple[int, EvaluationMetrics]:
    """Select Baseline B's cluster-size threshold using validation only."""
    best_threshold = 2
    best_metrics: EvaluationMetrics | None = None
    for threshold in range(2, 9):
        baseline = BaselineBGraphHeuristicOnly(cluster_threshold=threshold)
        predictions = [
            1 if baseline.predict_is_abuse(row, dataset.feature_names) else 0
            for row in dataset.feature_rows
        ]
        metrics = compute_binary_metrics(
            y_true=list(dataset.labels),
            y_pred=predictions,
        )
        if best_metrics is None:
            best_threshold, best_metrics = threshold, metrics
            continue
        candidate_key = (
            metrics.f1_score,
            metrics.precision,
            metrics.recall,
            -abs(threshold - 2),
        )
        best_key = (
            best_metrics.f1_score,
            best_metrics.precision,
            best_metrics.recall,
            -abs(best_threshold - 2),
        )
        if candidate_key > best_key:
            best_threshold, best_metrics = threshold, metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _loss_metrics(
    amounts: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float | int]:
    """Report prevented abuse exposure and legitimate review cost separately."""
    loss_prevented = sum(
        amount
        for amount, actual, predicted in zip(amounts, y_true, y_pred)
        if actual == 1 and predicted == 1
    )
    false_positive_amount = sum(
        amount
        for amount, actual, predicted in zip(amounts, y_true, y_pred)
        if actual == 0 and predicted == 1
    )
    flagged_amount = sum(
        amount
        for amount, predicted in zip(amounts, y_pred)
        if predicted == 1
    )
    return {
        "loss_prevented_paise": loss_prevented,
        "loss_prevented_inr": round(loss_prevented / 100, 2),
        "false_positive_flagged_amount_paise": false_positive_amount,
        "false_positive_flagged_amount_inr": round(false_positive_amount / 100, 2),
        "total_flagged_amount_paise": flagged_amount,
        "total_flagged_amount_inr": round(flagged_amount / 100, 2),
    }


def _summary(
    metrics: EvaluationMetrics,
    *,
    amounts: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    operating_threshold: float | int | None,
) -> dict[str, float | int | None]:
    cm = metrics.confusion_matrix
    result: dict[str, float | int | None] = {
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1_score": round(metrics.f1_score, 4),
        "accuracy": round(metrics.accuracy, 4),
        "true_positive": cm.true_positive,
        "true_negative": cm.true_negative,
        "false_positive": cm.false_positive,
        "false_negative": cm.false_negative,
        "review_volume": cm.true_positive + cm.false_positive,
        "operating_threshold": operating_threshold,
    }
    result.update(_loss_metrics(amounts, y_true, y_pred))
    return result


def run_evaluation_suite(output_path: Path | None = None) -> dict:
    """Run the full train/validation/held-out comparative benchmark."""
    print("================================================================")
    print(" REFUND SENTINEL: COMPARATIVE BASELINE EVALUATION")
    print("================================================================")

    print("\n[1/4] Generating train, validation, and held-out partitions...")
    train_rows_raw, train_summary = generate_partition(
        num_background_customers=100,
        include_held_out=False,
        seed=42,
    )
    validation_rows_raw, validation_summary = generate_partition(
        num_background_customers=50,
        include_held_out=False,
        seed=101,
    )
    test_rows_raw, test_summary = generate_partition(
        num_background_customers=50,
        include_held_out=True,
        seed=999,
    )

    feature_names = list(train_rows_raw[0]["feature_names"])
    train_dataset = _dataset_from_rows(train_rows_raw, feature_names)
    validation_dataset = _dataset_from_rows(validation_rows_raw, feature_names)
    test_dataset = _dataset_from_rows(test_rows_raw, feature_names)

    print(
        f"Train samples:      {len(train_rows_raw)} "
        f"(Abuse: {train_summary['abuse_count']}, Legit: {train_summary['legit_count']})"
    )
    print(
        f"Validation samples: {len(validation_rows_raw)} "
        f"(Abuse: {validation_summary['abuse_count']}, Legit: {validation_summary['legit_count']})"
    )
    print(
        f"Held-out samples:   {len(test_rows_raw)} "
        f"(Abuse: {test_summary['abuse_count']}, Legit: {test_summary['legit_count']})"
    )

    print("\n[2/4] Training benchmark models...")
    baseline_a = BaselineAIndividualOnly()
    baseline_a.fit(train_dataset)
    baseline_c = BaselineCFullSystem()
    baseline_c.fit(train_dataset)

    print("\n[3/4] Selecting operating thresholds on validation data...")
    validation_prob_a = [
        baseline_a.predict_proba(row)
        for row in validation_dataset.feature_rows
    ]
    threshold_a, validation_metrics_a = _select_probability_threshold(
        validation_prob_a,
        list(validation_dataset.labels),
    )

    validation_prob_c = [
        baseline_c.predict_proba(row)
        for row in validation_dataset.feature_rows
    ]
    threshold_c, validation_metrics_c = _select_probability_threshold(
        validation_prob_c,
        list(validation_dataset.labels),
    )

    structural_threshold, validation_metrics_b = _select_structural_threshold(
        validation_dataset
    )
    baseline_b = BaselineBGraphHeuristicOnly(
        cluster_threshold=structural_threshold
    )

    print(f"Baseline A probability threshold: {threshold_a:.2f}")
    print(f"Baseline B cluster-size threshold: {structural_threshold}")
    print(f"Baseline C probability threshold: {threshold_c:.2f}")

    print("\n[4/4] Evaluating once on unseen held-out scenario families...")
    y_true = list(test_dataset.labels)
    amounts = [r["requested_amount_paise"] for r in test_rows_raw]

    y_pred_a = [
        1 if baseline_a.predict_proba(row) >= threshold_a else 0
        for row in test_dataset.feature_rows
    ]
    metrics_a = compute_binary_metrics(y_true=y_true, y_pred=y_pred_a)

    y_pred_b = [
        1 if baseline_b.predict_is_abuse(row, feature_names) else 0
        for row in test_dataset.feature_rows
    ]
    metrics_b = compute_binary_metrics(y_true=y_true, y_pred=y_pred_b)

    y_pred_c = [
        1 if baseline_c.predict_proba(row) >= threshold_c else 0
        for row in test_dataset.feature_rows
    ]
    metrics_c = compute_binary_metrics(y_true=y_true, y_pred=y_pred_c)

    # Per-scenario recall for abuse and false-positive rate for legitimate cases.
    scenario_metrics: dict[str, dict[str, float | int]] = {}
    for scenario in sorted({r["scenario"] for r in test_rows_raw}):
        indices = [
            index
            for index, row in enumerate(test_rows_raw)
            if row["scenario"] == scenario
        ]
        if not indices:
            continue
        labels = [y_true[i] for i in indices]
        preds_a = [y_pred_a[i] for i in indices]
        preds_b = [y_pred_b[i] for i in indices]
        preds_c = [y_pred_c[i] for i in indices]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count

        def _scenario_row(predictions: Sequence[int]) -> dict[str, float]:
            if positive_count:
                recall = sum(
                    1 for actual, predicted in zip(labels, predictions)
                    if actual == 1 and predicted == 1
                ) / positive_count
                return {"recall": round(recall, 4)}
            fp_rate = sum(
                1 for actual, predicted in zip(labels, predictions)
                if actual == 0 and predicted == 1
            ) / max(1, negative_count)
            return {"false_positive_rate": round(fp_rate, 4)}

        scenario_metrics[scenario] = {
            "total_cases": len(indices),
            "abuse_cases": positive_count,
            "legitimate_cases": negative_count,
            "baseline_a": _scenario_row(preds_a),
            "baseline_b": _scenario_row(preds_b),
            "baseline_c": _scenario_row(preds_c),
        }

    results = {
        "summary": {
            "baseline_a_individual_only": _summary(
                metrics_a,
                amounts=amounts,
                y_true=y_true,
                y_pred=y_pred_a,
                operating_threshold=threshold_a,
            ),
            "baseline_b_graph_structural_only": _summary(
                metrics_b,
                amounts=amounts,
                y_true=y_true,
                y_pred=y_pred_b,
                operating_threshold=structural_threshold,
            ),
            "baseline_c_full_multi_signal": _summary(
                metrics_c,
                amounts=amounts,
                y_true=y_true,
                y_pred=y_pred_c,
                operating_threshold=threshold_c,
            ),
        },
        "validation_selection": {
            "baseline_a": {
                "selected_threshold": threshold_a,
                "f1_score": round(validation_metrics_a.f1_score, 4),
            },
            "baseline_b": {
                "selected_cluster_threshold": structural_threshold,
                "f1_score": round(validation_metrics_b.f1_score, 4),
            },
            "baseline_c": {
                "selected_threshold": threshold_c,
                "f1_score": round(validation_metrics_c.f1_score, 4),
            },
        },
        "evaluation_protocol": {
            "train_scenarios": ["AS01", "AS02", "LL01"],
            "validation_scenarios": ["AS01", "AS02", "LL01"],
            "heldout_test_scenarios": ["AS03", "AS04", "LL02", "LL03"],
            "heldout_design": (
                "AS03 tests structural coordination detection; AS04 tests "
                "isolated behavioral abuse; LL03 is a legitimate shared-"
                "household structural lookalike."
            ),
            "background_population_present_in_all_partitions": True,
            "model_selection_note": (
                "Probability and structural thresholds are selected using only "
                "the validation partition. Held-out labels are used exactly once "
                "for final reporting."
            ),
        },
        "total_test_exposure_inr": round(
            sum(amount for amount, label in zip(amounts, y_true) if label == 1)
            / 100,
            2,
        ),
        "per_scenario_metrics": scenario_metrics,
    }

    print("\n" + "=" * 116)
    print(
        f"{'Model / Baseline':<38} | {'Precision':<9} | {'Recall':<9} | "
        f"{'F1':<9} | {'FP':<4} | {'FN':<4} | {'Loss Blocked (INR)'}"
    )
    print("-" * 116)
    for label, metrics in (
        ("Baseline A (Individual-Only)", metrics_a),
        ("Baseline B (Graph Structural Only)", metrics_b),
        ("Baseline C (Full Multi-Signal System)", metrics_c),
    ):
        key = {
            "Baseline A (Individual-Only)": "baseline_a_individual_only",
            "Baseline B (Graph Structural Only)": "baseline_b_graph_structural_only",
            "Baseline C (Full Multi-Signal System)": "baseline_c_full_multi_signal",
        }[label]
        summary = results["summary"][key]
        print(
            f"{label:<38} | {metrics.precision:<9.4f} | {metrics.recall:<9.4f} | "
            f"{metrics.f1_score:<9.4f} | {metrics.confusion_matrix.false_positive:<4} | "
            f"{metrics.confusion_matrix.false_negative:<4} | "
            f"INR {summary['loss_prevented_inr']:,.2f}"
        )
    print("=" * 116)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults successfully written to: {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation across Refund Sentinel baselines."
    )
    parser.add_argument("--output", type=str, default="data/results.json")
    args = parser.parse_args()
    run_evaluation_suite(Path(args.output))


if __name__ == "__main__":
    main()
