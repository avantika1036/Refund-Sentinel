"""Run the reproducible Refund Sentinel comparative benchmark.

Protocol:
- train: known mechanism families
- validation: independent seeds of known families; thresholds selected here
- held-out test: unseen scenario families and a larger legitimate structural
  lookalike used exactly once for final reporting

The benchmark reports classification quality plus explicit false-positive cost
sensitivity. Cost assumptions are configurable and are clearly separated from
measured model performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.app.ml.baselines import (
    BaselineAIndividualOnly,
    BaselineBGraphHeuristicOnly,
    BaselineCFullSystem,
)
from backend.app.ml.dataset import MLDataset, create_dataset
from backend.app.ml.evaluation import EvaluationMetrics, compute_binary_metrics
from backend.app.ml.persistence import PersistedModelBundle, save_model_bundle
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
    """Select a threshold on validation data only.

    F1 is primary, then precision, recall, and proximity to 0.5 as tie-breakers.
    The held-out test labels are never consulted here.
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
    """Select Baseline B's topology threshold on validation data only."""
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


def _financial_metrics(
    amounts_paise: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    fp_cost_per_case_inr: float,
) -> dict[str, float | int]:
    """Report measured exposure capture and configurable review costs.

    Amount capture is intentionally named "captured", not "prevented": the
    offline classifier did not actually stop any money from leaving the merchant.
    """
    captured_abuse_paise = sum(
        amount
        for amount, actual, predicted in zip(amounts_paise, y_true, y_pred)
        if actual == 1 and predicted == 1
    )
    missed_abuse_paise = sum(
        amount
        for amount, actual, predicted in zip(amounts_paise, y_true, y_pred)
        if actual == 1 and predicted == 0
    )
    flagged_legitimate_paise = sum(
        amount
        for amount, actual, predicted in zip(amounts_paise, y_true, y_pred)
        if actual == 0 and predicted == 1
    )
    abuse_total_paise = sum(
        amount for amount, actual in zip(amounts_paise, y_true) if actual == 1
    )
    fp_count = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 0 and predicted == 1
    )
    review_volume = sum(y_pred)

    return {
        "abuse_exposure_captured_paise": captured_abuse_paise,
        "abuse_exposure_captured_inr": round(captured_abuse_paise / 100, 2),
        "missed_abuse_exposure_paise": missed_abuse_paise,
        "missed_abuse_exposure_inr": round(missed_abuse_paise / 100, 2),
        "total_abuse_exposure_paise": abuse_total_paise,
        "total_abuse_exposure_inr": round(abuse_total_paise / 100, 2),
        "abuse_exposure_capture_rate": round(
            captured_abuse_paise / abuse_total_paise if abuse_total_paise else 0.0,
            4,
        ),
        "false_positive_flagged_amount_paise": flagged_legitimate_paise,
        "false_positive_flagged_amount_inr": round(flagged_legitimate_paise / 100, 2),
        "false_positive_count": fp_count,
        "review_volume": review_volume,
        "assumed_fp_cost_per_case_inr": fp_cost_per_case_inr,
        "false_positive_operational_cost_inr": round(
            fp_count * fp_cost_per_case_inr,
            2,
        ),
    }


def _summary(
    metrics: EvaluationMetrics,
    *,
    amounts: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    operating_threshold: float | int,
    fp_cost_per_case_inr: float,
) -> dict[str, float | int]:
    cm = metrics.confusion_matrix
    result: dict[str, float | int] = {
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1_score": round(metrics.f1_score, 4),
        "accuracy": round(metrics.accuracy, 4),
        "true_positive": cm.true_positive,
        "true_negative": cm.true_negative,
        "false_positive": cm.false_positive,
        "false_negative": cm.false_negative,
        "false_positive_rate": round(
            cm.false_positive / max(1, cm.false_positive + cm.true_negative), 4
        ),
        "false_negative_rate": round(
            cm.false_negative / max(1, cm.false_negative + cm.true_positive), 4
        ),
        "operating_threshold": operating_threshold,
    }
    result.update(
        _financial_metrics(
            amounts,
            y_true,
            y_pred,
            fp_cost_per_case_inr=fp_cost_per_case_inr,
        )
    )
    return result


def _scenario_row(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count:
        return {
            "recall": round(
                sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
                / positive_count,
                4,
            )
        }
    return {
        "false_positive_rate": round(
            sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 1)
            / max(1, negative_count),
            4,
        )
    }


def _build_cost_sensitivity(
    amounts: Sequence[int],
    y_true: Sequence[int],
    predictions_by_model: dict[str, Sequence[int]],
    cost_points: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Show how conclusions change under several explicit FP cost assumptions."""
    output: dict[str, dict[str, float]] = {}
    for model_name, predictions in predictions_by_model.items():
        fp_count = sum(
            1 for actual, predicted in zip(y_true, predictions)
            if actual == 0 and predicted == 1
        )
        output[model_name] = {
            f"fp_cost_{cost:g}_inr": round(fp_count * cost, 2)
            for cost in cost_points
        }
    return output


def run_evaluation_suite(
    output_path: Path | None = None,
    *,
    fp_cost_points_inr: Sequence[float] = (50.0, 200.0, 500.0),
) -> dict:
    """Run training, validation threshold selection, and one final held-out test."""
    print("=" * 72)
    print(" REFUND SENTINEL: COMPARATIVE BASELINE EVALUATION")
    print("=" * 72)

    print("\n[1/4] Generating independent train/validation/held-out partitions...")
    train_rows_raw, train_summary = generate_partition(100, include_held_out=False, seed=42)
    validation_rows_raw, validation_summary = generate_partition(50, include_held_out=False, seed=101)
    test_rows_raw, test_summary = generate_partition(50, include_held_out=True, seed=999)

    feature_names = list(train_rows_raw[0]["feature_names"])
    train_dataset = _dataset_from_rows(train_rows_raw, feature_names)
    validation_dataset = _dataset_from_rows(validation_rows_raw, feature_names)
    test_dataset = _dataset_from_rows(test_rows_raw, feature_names)

    print(f"Train:      {len(train_rows_raw)} rows ({train_summary['abuse_count']} abuse / {train_summary['legit_count']} legit)")
    print(f"Validation: {len(validation_rows_raw)} rows ({validation_summary['abuse_count']} abuse / {validation_summary['legit_count']} legit)")
    print(f"Held-out:   {len(test_rows_raw)} rows ({test_summary['abuse_count']} abuse / {test_summary['legit_count']} legit)")

    if test_summary["abuse_count"] < 20 or test_summary["legit_count"] < 20:
        raise RuntimeError("Held-out benchmark is too small for a credible comparative report")

    print("\n[2/4] Fitting Baseline A and Baseline C on training only...")
    baseline_a = BaselineAIndividualOnly()
    baseline_a.fit(train_dataset)
    baseline_c = BaselineCFullSystem()
    baseline_c.fit(train_dataset)
    if baseline_c.model is None or baseline_c.preprocessor is None:
        raise RuntimeError("Baseline C did not produce a deployable model bundle")
    model_path = PROJECT_ROOT / "backend" / "app" / "models" / "risk_model.json"
    save_model_bundle(
        PersistedModelBundle(
            model=baseline_c.model,
            preprocessor=baseline_c.preprocessor,
        ),
        model_path,
    )
    print(f"Deployable Baseline C artifact: {model_path}")

    print("\n[3/4] Selecting thresholds using validation only...")
    validation_prob_a = [baseline_a.predict_proba(row) for row in validation_dataset.feature_rows]
    threshold_a, validation_metrics_a = _select_probability_threshold(
        validation_prob_a, list(validation_dataset.labels)
    )
    validation_prob_c = [baseline_c.predict_proba(row) for row in validation_dataset.feature_rows]
    threshold_c, validation_metrics_c = _select_probability_threshold(
        validation_prob_c, list(validation_dataset.labels)
    )
    structural_threshold, validation_metrics_b = _select_structural_threshold(validation_dataset)
    baseline_b = BaselineBGraphHeuristicOnly(cluster_threshold=structural_threshold)

    print(f"Baseline A threshold: {threshold_a:.2f}")
    print(f"Baseline B cluster threshold: {structural_threshold}")
    print(f"Baseline C threshold: {threshold_c:.2f}")

    print("\n[4/4] Evaluating exactly once on held-out scenario families...")
    y_true = list(test_dataset.labels)
    amounts = [int(r["requested_amount_paise"]) for r in test_rows_raw]

    y_pred_a = [1 if baseline_a.predict_proba(row) >= threshold_a else 0 for row in test_dataset.feature_rows]
    y_pred_b = [1 if baseline_b.predict_is_abuse(row, feature_names) else 0 for row in test_dataset.feature_rows]
    y_pred_c = [1 if baseline_c.predict_proba(row) >= threshold_c else 0 for row in test_dataset.feature_rows]

    metrics_a = compute_binary_metrics(y_true=y_true, y_pred=y_pred_a)
    metrics_b = compute_binary_metrics(y_true=y_true, y_pred=y_pred_b)
    metrics_c = compute_binary_metrics(y_true=y_true, y_pred=y_pred_c)

    fp_reference_cost = float(fp_cost_points_inr[1]) if len(fp_cost_points_inr) > 1 else float(fp_cost_points_inr[0])

    scenario_metrics: dict[str, dict[str, float | int]] = {}
    for scenario in sorted({r["scenario"] for r in test_rows_raw}):
        indices = [i for i, row in enumerate(test_rows_raw) if row["scenario"] == scenario]
        labels = [y_true[i] for i in indices]
        scenario_metrics[scenario] = {
            "total_cases": len(indices),
            "abuse_cases": sum(labels),
            "legitimate_cases": len(labels) - sum(labels),
            "baseline_a_recall_or_fpr": _scenario_row(labels, [y_pred_a[i] for i in indices]),
            "baseline_b_recall_or_fpr": _scenario_row(labels, [y_pred_b[i] for i in indices]),
            "baseline_c_recall_or_fpr": _scenario_row(labels, [y_pred_c[i] for i in indices]),
        }

    results = {
        "summary": {
            "baseline_a_individual_only": _summary(
                metrics_a, amounts=amounts, y_true=y_true, y_pred=y_pred_a,
                operating_threshold=threshold_a, fp_cost_per_case_inr=fp_reference_cost,
            ),
            "baseline_b_graph_structural_only": _summary(
                metrics_b, amounts=amounts, y_true=y_true, y_pred=y_pred_b,
                operating_threshold=structural_threshold, fp_cost_per_case_inr=fp_reference_cost,
            ),
            "baseline_c_full_multi_signal": _summary(
                metrics_c, amounts=amounts, y_true=y_true, y_pred=y_pred_c,
                operating_threshold=threshold_c, fp_cost_per_case_inr=fp_reference_cost,
            ),
        },
        "validation_selection": {
            "baseline_a": {"selected_threshold": threshold_a, "f1_score": round(validation_metrics_a.f1_score, 4)},
            "baseline_b": {"selected_cluster_threshold": structural_threshold, "f1_score": round(validation_metrics_b.f1_score, 4)},
            "baseline_c": {"selected_threshold": threshold_c, "f1_score": round(validation_metrics_c.f1_score, 4)},
        },
        "evaluation_protocol": {
            "train_scenarios": ["AS01", "AS02", "LL01"],
            "validation_scenarios": ["AS01", "AS02", "LL01"],
            "heldout_test_scenarios": ["AS03", "AS04", "LL02", "LL03"],
            "heldout_design": (
                "AS03 tests an unseen shared-attribute coordination pattern; AS04 tests "
                "isolated high-velocity abuse; LL02 tests established legitimate high activity; "
                "LL03 is a larger legitimate shared-household structural lookalike."
            ),
            "threshold_selection": "Validation only; held-out labels are never used for threshold selection.",
            "point_in_time_features": True,
            "model_selection_note": "The held-out set is used once for final comparative reporting.",
        },
        "false_positive_cost_sensitivity": _build_cost_sensitivity(
            amounts, y_true,
            {
                "baseline_a_individual_only": y_pred_a,
                "baseline_b_graph_structural_only": y_pred_b,
                "baseline_c_full_multi_signal": y_pred_c,
            },
            fp_cost_points_inr,
        ),
        "total_test_abuse_exposure_inr": round(
            sum(amount for amount, label in zip(amounts, y_true) if label == 1) / 100,
            2,
        ),
        "per_scenario_metrics": scenario_metrics,
    }

    print("\n" + "=" * 100)
    print(f"{'Model':<35} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'FP':<4} | {'FN':<4} | {'Captured abuse ₹'}")
    print("-" * 100)
    for label, metrics, key in (
        ("Baseline A (Individual-Only)", metrics_a, "baseline_a_individual_only"),
        ("Baseline B (Graph Structural Only)", metrics_b, "baseline_b_graph_structural_only"),
        ("Baseline C (Full Multi-Signal)", metrics_c, "baseline_c_full_multi_signal"),
    ):
        summary = results["summary"][key]
        print(
            f"{label:<35} | {metrics.precision:<9.4f} | {metrics.recall:<9.4f} | "
            f"{metrics.f1_score:<9.4f} | {metrics.confusion_matrix.false_positive:<4} | "
            f"{metrics.confusion_matrix.false_negative:<4} | "
            f"₹{summary['abuse_exposure_captured_inr']:,.2f}"
        )
    print("=" * 100)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to: {output_path.resolve()}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/results.json"))
    parser.add_argument(
        "--fp-cost-points",
        type=float,
        nargs="+",
        default=[50.0, 200.0, 500.0],
        help="Explicit sensitivity-analysis assumptions in INR per false positive.",
    )
    args = parser.parse_args()
    run_evaluation_suite(args.output, fp_cost_points_inr=args.fp_cost_points)


if __name__ == "__main__":
    main()
