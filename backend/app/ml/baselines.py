"""Baseline models for comparative evaluation of Refund Sentinel.

Implements the three core benchmark baselines from Section 10 of DOC.MD:
- Baseline A: Individual-Only features (Transaction velocity & history, no graph/cluster signals)
- Baseline B: Graph Structural Only (Pure topology, no behavioral rules, no ML)
- Baseline C: Full Multi-Signal System (Individual + Graph + Financial State + ML)
"""

from __future__ import annotations

from typing import Sequence

from backend.app.ml.dataset import MLDataset, create_dataset
from backend.app.ml.model import LogisticRiskModel, train_logistic_risk_model
from backend.app.ml.preprocessing import MLPreprocessor, fit_preprocessor


class BaselineAIndividualOnly:
    """Baseline A: Trained strictly on individual transaction and refund features."""

    def __init__(self) -> None:
        self.model: LogisticRiskModel | None = None
        self.preprocessor: MLPreprocessor | None = None
        self.feature_indices: list[int] = []

    def fit(self, dataset: MLDataset) -> None:
        """Fit model only using individual_* features."""
        self.feature_indices = [
            i for i, name in enumerate(dataset.feature_names)
            if name.startswith("individual_")
        ]

        if not self.feature_indices:
            raise ValueError("No individual features found in dataset")

        individual_names = [dataset.feature_names[i] for i in self.feature_indices]
        individual_rows = [
            [row[i] for i in self.feature_indices]
            for row in dataset.feature_rows
        ]

        individual_dataset = create_dataset(
            feature_names=individual_names,
            feature_rows=individual_rows,
            labels=list(dataset.labels),
        )
        self.preprocessor = fit_preprocessor(
            feature_names=individual_dataset.feature_names,
            feature_rows=individual_dataset.feature_rows,
        )
        processed_rows = self.preprocessor.transform(
            feature_names=individual_dataset.feature_names,
            feature_rows=individual_dataset.feature_rows,
        )
        processed_dataset = create_dataset(
            feature_names=individual_dataset.feature_names,
            feature_rows=processed_rows,
            labels=individual_dataset.labels,
        )
        self.model = train_logistic_risk_model(processed_dataset).model

    def predict_proba(self, feature_row: Sequence[float]) -> float:
        """Predict probability using individual features only."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Baseline A model is not fitted")
        filtered_values = [feature_row[i] for i in self.feature_indices]
        transformed = self.preprocessor.transform(
            feature_names=self.model.feature_names,
            feature_rows=[filtered_values],
        )
        return self.model.predict_probability(transformed[0])


class BaselineBGraphHeuristicOnly:
    """Baseline B: pure structural-graph heuristic with no ML or behavior rules.

    This baseline intentionally uses only topology-derived features. It must
    not consume refund velocity, timing, active-refund fractions, deterministic
    rule outputs, or any other behavioral/financial signal.
    """

    def __init__(self, cluster_threshold: int = 2) -> None:
        self.cluster_threshold = cluster_threshold

    def predict_is_abuse(
        self,
        feature_row: Sequence[float],
        feature_names: Sequence[str],
        *,
        triggered_rule_count: int = 0,
    ) -> bool:
        """Flag a structurally connected multi-account cluster.

        ``triggered_rule_count`` is accepted only for backwards-compatible
        callers and is deliberately ignored: deterministic rule evidence is
        not graph topology and therefore does not belong in this baseline.
        """
        feat_dict = dict(zip(feature_names, feature_row))
        cluster_size = feat_dict.get("cluster_cluster_size", 1.0)
        shared_attribute_types = feat_dict.get(
            "relationship_shared_attribute_type_count",
            0.0,
        )

        return bool(
            cluster_size >= self.cluster_threshold
            and shared_attribute_types >= 1.0
        )


class BaselineCFullSystem:
    """Baseline C: Full Multi-Signal ML system combining all feature dimensions."""

    def __init__(self) -> None:
        self.model: LogisticRiskModel | None = None
        self.preprocessor: MLPreprocessor | None = None

    def fit(self, dataset: MLDataset) -> None:
        """Fit model on complete feature space."""
        self.preprocessor = fit_preprocessor(
            feature_names=dataset.feature_names,
            feature_rows=dataset.feature_rows,
        )
        processed_rows = self.preprocessor.transform(
            feature_names=dataset.feature_names,
            feature_rows=dataset.feature_rows,
        )
        processed_dataset = create_dataset(
            feature_names=dataset.feature_names,
            feature_rows=processed_rows,
            labels=dataset.labels,
        )
        self.model = train_logistic_risk_model(processed_dataset).model

    def predict_proba(self, feature_row: Sequence[float]) -> float:
        """Predict probability using all features."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Baseline C model is not fitted")
        transformed = self.preprocessor.transform(
            feature_names=self.model.feature_names,
            feature_rows=[feature_row],
        )
        return self.model.predict_probability(transformed[0])
