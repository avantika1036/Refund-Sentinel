"""Ground-truth labels for simulator-generated scenarios.

Ground truth is simulation metadata, NOT production features.
It identifies which entities/events belong to which scenarios and
whether they represent abuse or legitimate behavior for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.domain.events import AnyDomainEvent
    from backend.app.domain.identifiers import CustomerId, PaymentId, RefundId


class ScenarioType(str, Enum):
    """Identifies the type of scenario for ground-truth labeling."""

    # Abuse scenarios
    AS01_DENSE_COORDINATED_REFUND_RING = "AS01_DENSE_COORDINATED_REFUND_RING"
    AS02_VELOCITY_REFUND_ABUSE = "AS02_VELOCITY_REFUND_ABUSE"
    AS03_SHARED_PAYMENT_DEVICE_RING = "AS03_SHARED_PAYMENT_DEVICE_RING"

    # Legitimate lookalike scenarios
    LL01_LEGITIMATE_FAMILY = "LL01_LEGITIMATE_FAMILY"
    LL02_FREQUENT_SHOPPER = "LL02_FREQUENT_SHOPPER"

    # Background population
    BACKGROUND = "BACKGROUND"


class LabelClassification(str, Enum):
    """Ground-truth classification for entities/events."""

    ABUSE = "abuse"
    LEGITIMATE = "legitimate"


@dataclass(frozen=True)
class GroundTruthLabel:
    """Ground-truth label for a simulator-generated entity or event.

    This is simulation metadata for evaluation, NOT a production feature.
    The scoring pipeline must NOT use ground-truth labels as input features.
    """

    scenario_type: ScenarioType
    classification: LabelClassification
    customer_id: CustomerId | None = None
    payment_id: PaymentId | None = None
    refund_id: RefundId | None = None
    event_id: object | None = None
    description: str = ""

    def is_abuse(self) -> bool:
        return self.classification == LabelClassification.ABUSE

    def is_legitimate(self) -> bool:
        return self.classification == LabelClassification.LEGITIMATE


@dataclass(frozen=True)
class SimulationOutput:
    """Complete output from a simulation run.

    Contains both the generated events and their ground-truth labels.
    """

    events: list[AnyDomainEvent] = field(default_factory=list)
    labels: list[GroundTruthLabel] = field(default_factory=list)
    seed: int = 0

    def get_labels_for_scenario(self, scenario_type: ScenarioType) -> list[GroundTruthLabel]:
        """Return all labels for a specific scenario type."""
        return [label for label in self.labels if label.scenario_type == scenario_type]

    def get_abuse_labels(self) -> list[GroundTruthLabel]:
        """Return all labels classified as abuse."""
        return [label for label in self.labels if label.is_abuse()]

    def get_legitimate_labels(self) -> list[GroundTruthLabel]:
        """Return all labels classified as legitimate."""
        return [label for label in self.labels if label.is_legitimate()]
