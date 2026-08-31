from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from backend.app.domain.enums import EventType
from backend.app.domain.identifiers import EventId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import UTCDateTime

if TYPE_CHECKING:
    from backend.app.domain.events import AnyDomainEvent
    from backend.app.finance.aggregates import OrderState, PaymentState, RefundState


class IngestionOutcome(str, Enum):
    RETAINED = "retained"
    PENDING = "pending"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class PendingReason(str, Enum):
    PREREQUISITE_NOT_YET_SEEN = "prerequisite_not_yet_seen"


@dataclass(frozen=True)
class RetainedEvent:
    """An event accepted into the canonical reconstruction store."""

    event: "AnyDomainEvent"
    retained_at: UTCDateTime
    canonical_position: int


@dataclass(frozen=True)
class PendingEvent:
    """A valid event waiting for a prerequisite event/state."""

    event: "AnyDomainEvent"
    received_at: UTCDateTime
    pending_reason: PendingReason


@dataclass
class IngestionRecord:
    """Audit record for exactly one submission at the ingestion boundary."""

    event_id: EventId
    event_type: EventType
    occurred_at: UTCDateTime
    received_at: UTCDateTime
    submission_ordinal: int
    payload_hash: str
    ingestion_outcome: IngestionOutcome
    reason_code: object | None = None
    detail: str | None = None
    retained: bool = False
    pending: bool = False
    promoted_at: UTCDateTime | None = None
    triggered_reconstruction: bool = False
    reconstruction_ordinal: int | None = None
    original_submission_ordinal: int | None = None


@dataclass(frozen=True)
class QuarantinedEvent:
    event: "AnyDomainEvent"
    quarantined_at: UTCDateTime
    reason_code: object
    detail: str | None = None


@dataclass(frozen=True)
class ReconstructionAnomaly:
    event_id: EventId
    event_type: EventType
    reason: str


@dataclass(frozen=True)
class ReconstructionSnapshot:
    """Complete state produced by a deterministic reconstruction pass."""

    payments: dict[PaymentId, "PaymentState"]
    refunds: dict[RefundId, "RefundState"]
    orders: dict[OrderId, "OrderState"]
    reconstruction_ordinal: int
    event_count: int
    anomalies: tuple[ReconstructionAnomaly, ...] = field(default_factory=tuple)
