from __future__ import annotations

from datetime import datetime, timezone

from backend.app.domain.enums import DataSource, EventType
from backend.app.domain.events import EventEnvelope, PaymentCreatedEvent, PaymentCreatedPayload
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.persistence.models import EventModel
from backend.app.persistence.reconstruction import ReconstructionService


def test_reconstruction_is_pure_and_reports_corrupt_rows(monkeypatch) -> None:
    event = PaymentCreatedEvent(
        envelope=EventEnvelope(event_id=EventId.generate(), event_type=EventType.PAYMENT_CREATED, occurred_at=UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc)), received_at=UTCDateTime(value=datetime(2024, 1, 1, tzinfo=timezone.utc)), source=DataSource.SIMULATOR),
        payload=PaymentCreatedPayload(payment_id=PaymentId.generate(), order_id=OrderId.generate(), merchant_id=MerchantId.generate(), customer_id=CustomerId.generate(), amount=Money.of_paise(100)),
    )
    row = EventModel(event_id=event.envelope.event_id.value, event_type=event.envelope.event_type.value, occurred_at=event.envelope.occurred_at.value, received_at=event.envelope.received_at.value, source=event.envelope.source.value, payload_hash="x" * 64, payload=event.model_dump(mode="json"), created_at=datetime.now(timezone.utc))
    original = row.payload.copy()
    monkeypatch.setattr("backend.app.persistence.reconstruction.EventRepository", lambda _: type("Repo", (), {"list_by_occurred_at": lambda self: [row]})())
    service = ReconstructionService(object())
    first, second = service.reconstruct(), service.reconstruct()
    assert first == second and first.event_count == 1 and row.payload == original

    row.payload = {"envelope": {"event_type": "invalid"}}
    corrupt = service.reconstruct()
    assert corrupt.event_count == 0 and len(corrupt.anomalies) == 1
