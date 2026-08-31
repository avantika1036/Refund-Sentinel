"""Abuse and legitimate scenario generators.

AS-01: Dense coordinated refund ring — multiple customers with shared structural
attributes and coordinated refund behavior.

LL-01: Legitimate family — shared structural attributes but natural lifecycle
variation, should NOT be treated as abuse.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from backend.app.config import settings
from backend.app.domain.enums import DataSource, EventType, RefundReasonCode
from backend.app.domain.events import (
    AnyDomainEvent,
    EventEnvelope,
    OrderCreatedEvent,
    OrderCreatedPayload,
    OrderDeliveredEvent,
    OrderDeliveredPayload,
    OrderShippedEvent,
    OrderShippedPayload,
    PaymentCapturedEvent,
    PaymentCapturedPayload,
    PaymentCreatedEvent,
    PaymentCreatedPayload,
    RefundCreatedEvent,
    RefundCreatedPayload,
    RefundProcessedEvent,
    RefundProcessedPayload,
    RefundRequestedEvent,
    RefundRequestedPayload,
)
from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    EventId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
)
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.simulator.labels import (
    GroundTruthLabel,
    LabelClassification,
    ScenarioType,
    SimulationOutput,
)

if TYPE_CHECKING:
    pass


class Scenario:
    """Base class for scenario generators."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed or settings.demo_seed
        self._rng = random.Random(self._seed)
        self._events: list[AnyDomainEvent] = []
        self._labels: list[GroundTruthLabel] = []

    def generate(self) -> SimulationOutput:
        """Generate the scenario events and labels."""
        raise NotImplementedError

    def _generate_event_id(self) -> EventId:
        """Generate a deterministic event ID."""
        import uuid
        seed_int = self._rng.randint(0, 2**32 - 1)
        return EventId(uuid.UUID(int=seed_int))


class AS01_DenseCoordinatedRefundRing(Scenario):
    """AS-01: Dense coordinated refund ring.

    Multiple distinct customers with shared structural attributes (device/address)
    and coordinated refund behavior. Represents a coordinated abuse pattern.
    """

    def generate(
        self,
        num_customers: int = 5,
        orders_per_customer: int = 3,
    ) -> SimulationOutput:
        """Generate AS-01 abuse scenario.

        Key characteristics:
        - Multiple customers share the same device_id and address_id
        - Coordinated refund timing (refunds requested in close succession)
        - Similar refund reasons (e.g., all DEFECTIVE)
        - High refund rate compared to background
        """
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 2, 1, tzinfo=timezone.utc)

        # Shared structural attributes for coordination signal
        shared_device_id = DeviceId.generate()
        shared_address_id = AddressId.generate()

        customers = []
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            customers.append(customer_id)

            for order_idx in range(orders_per_customer):
                self._generate_coordinated_lifecycle(
                    customer_id,
                    merchant_id,
                    shared_device_id,
                    shared_address_id,
                    base_time,
                    customer_idx,
                    order_idx,
                )

        return SimulationOutput(
            events=self._events,
            labels=self._labels,
            seed=self._seed,
        )

    def _generate_coordinated_lifecycle(
        self,
        customer_id: CustomerId,
        merchant_id: MerchantId,
        shared_device_id: DeviceId,
        shared_address_id: AddressId,
        base_time: datetime,
        customer_idx: int,
        order_idx: int,
    ) -> None:
        """Generate a coordinated lifecycle with shared attributes and timing."""
        # Coordinated timing - all orders happen in a tight window
        time_offset_hours = customer_idx * 2 + order_idx * 4
        order_time = base_time + timedelta(hours=time_offset_hours)
        order_time = UTCDateTime(value=order_time)

        order_id = OrderId.generate()
        amount_paise = random.randint(5000, 30000)  # ₹50 to ₹300
        amount = Money.of_paise(amount_paise)

        # Order created
        order_created = OrderCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_CREATED,
                occurred_at=order_time,
                received_at=order_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderCreatedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                shipping_address_id=shared_address_id,
            ),
        )
        self._events.append(order_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                event_id=order_created.envelope.event_id,
                description="AS-01 coordinated order with shared address",
            )
        )

        # Payment created with shared device
        payment_id = PaymentId.generate()
        payment_time = UTCDateTime(value=order_time.value + timedelta(minutes=10))

        payment_created = PaymentCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CREATED,
                occurred_at=payment_time,
                received_at=payment_time,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCreatedPayload(
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                device_id=shared_device_id,
            ),
        )
        self._events.append(payment_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_created.envelope.event_id,
                description="AS-01 coordinated payment with shared device",
            )
        )

        # Payment captured
        capture_time = UTCDateTime(value=payment_time.value + timedelta(minutes=5))

        payment_captured = PaymentCapturedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CAPTURED,
                occurred_at=capture_time,
                received_at=capture_time,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCapturedPayload(
                payment_id=payment_id,
                merchant_id=merchant_id,
                captured_amount=amount,
                captured_at=capture_time,
            ),
        )
        self._events.append(payment_captured)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_captured.envelope.event_id,
                description="AS-01 coordinated payment captured",
            )
        )

        # Order shipped
        ship_time = UTCDateTime(value=capture_time.value + timedelta(hours=6))

        order_shipped = OrderShippedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_SHIPPED,
                occurred_at=ship_time,
                received_at=ship_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderShippedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                shipped_at=ship_time,
            ),
        )
        self._events.append(order_shipped)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                event_id=order_shipped.envelope.event_id,
                description="AS-01 coordinated order shipped",
            )
        )

        # Order delivered
        deliver_time = UTCDateTime(value=ship_time.value + timedelta(days=2))

        order_delivered = OrderDeliveredEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_DELIVERED,
                occurred_at=deliver_time,
                received_at=deliver_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderDeliveredPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                delivered_at=deliver_time,
            ),
        )
        self._events.append(order_delivered)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                event_id=order_delivered.envelope.event_id,
                description="AS-01 coordinated order delivered",
            )
        )

        # Coordinated refund request - all customers request refunds in close succession
        refund_id = RefundId.generate()
        refund_paise = int(amount.amount_paise * 0.8)  # 80% refund
        refund_amount = Money.of_paise(refund_paise)

        # Coordinated timing: refunds requested within hours of each other
        refund_request_delay = customer_idx * 3  # 3-hour intervals
        refund_request_time = UTCDateTime(value=deliver_time.value + timedelta(hours=refund_request_delay))

        # Same reason code for coordination signal
        refund_requested = RefundRequestedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_REQUESTED,
                occurred_at=refund_request_time,
                received_at=refund_request_time,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundRequestedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=refund_amount,
                reason_code=RefundReasonCode.DEFECTIVE,  # Coordinated reason
            ),
        )
        self._events.append(refund_requested)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_requested.envelope.event_id,
                description="AS-01 coordinated refund request (DEFECTIVE)",
            )
        )

        # Refund created
        created_time = UTCDateTime(value=refund_request_time.value + timedelta(minutes=30))

        refund_created = RefundCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_CREATED,
                occurred_at=created_time,
                received_at=created_time,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundCreatedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                created_at=created_time,
            ),
        )
        self._events.append(refund_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_created.envelope.event_id,
                description="AS-01 coordinated refund created",
            )
        )

        # Refund processed
        processed_time = UTCDateTime(value=created_time.value + timedelta(days=1))

        refund_processed = RefundProcessedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_PROCESSED,
                occurred_at=processed_time,
                received_at=processed_time,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundProcessedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                processed_at=processed_time,
                processed_amount=refund_amount,
            ),
        )
        self._events.append(refund_processed)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING,
                classification=LabelClassification.ABUSE,
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_processed.envelope.event_id,
                description="AS-01 coordinated refund processed",
            )
        )


class LL01_LegitimateFamily(Scenario):
    """LL-01: Legitimate family scenario.

    Family members share structural attributes (address/device) but have
    natural lifecycle variation. Should NOT be treated as abuse.
    """

    def generate(
        self,
        num_family_members: int = 4,
        orders_per_member: int = 2,
    ) -> SimulationOutput:
        """Generate LL-01 legitimate family scenario.

        Key characteristics:
        - Family members share the same address_id and device_id
        - Natural variation in refund timing (not coordinated)
        - Varied refund reasons (not all the same)
        - Lower refund rate than AS-01
        - Legitimate lifecycle patterns
        """
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 3, 1, tzinfo=timezone.utc)

        # Shared family address and device
        family_address_id = AddressId.generate()
        family_device_id = DeviceId.generate()

        family_members = []
        for member_idx in range(num_family_members):
            customer_id = CustomerId.generate()
            family_members.append(customer_id)

            for order_idx in range(orders_per_member):
                self._generate_family_lifecycle(
                    customer_id,
                    merchant_id,
                    family_address_id,
                    family_device_id,
                    base_time,
                    member_idx,
                    order_idx,
                )

        return SimulationOutput(
            events=self._events,
            labels=self._labels,
            seed=self._seed,
        )

    def _generate_family_lifecycle(
        self,
        customer_id: CustomerId,
        merchant_id: MerchantId,
        family_address_id: AddressId,
        family_device_id: DeviceId,
        base_time: datetime,
        member_idx: int,
        order_idx: int,
    ) -> None:
        """Generate a family lifecycle with natural variation."""
        # Natural timing variation - spread over weeks, not hours
        time_offset_days = member_idx * 7 + order_idx * 14
        order_time = base_time + timedelta(days=time_offset_days)
        order_time = UTCDateTime(value=order_time)

        order_id = OrderId.generate()
        amount_paise = random.randint(3000, 40000)  # ₹30 to ₹400
        amount = Money.of_paise(amount_paise)

        # Order created
        order_created = OrderCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_CREATED,
                occurred_at=order_time,
                received_at=order_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderCreatedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                shipping_address_id=family_address_id,
            ),
        )
        self._events.append(order_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                classification=LabelClassification.LEGITIMATE,
                customer_id=customer_id,
                event_id=order_created.envelope.event_id,
                description="LL-01 family order with shared address",
            )
        )

        # Payment created with family device
        payment_id = PaymentId.generate()
        payment_time = UTCDateTime(value=order_time.value + timedelta(minutes=random.randint(5, 45)))

        payment_created = PaymentCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CREATED,
                occurred_at=payment_time,
                received_at=payment_time,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCreatedPayload(
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                device_id=family_device_id,
            ),
        )
        self._events.append(payment_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                classification=LabelClassification.LEGITIMATE,
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_created.envelope.event_id,
                description="LL-01 family payment with shared device",
            )
        )

        # Payment captured
        capture_time = UTCDateTime(value=payment_time.value + timedelta(minutes=random.randint(2, 15)))

        payment_captured = PaymentCapturedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CAPTURED,
                occurred_at=capture_time,
                received_at=capture_time,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCapturedPayload(
                payment_id=payment_id,
                merchant_id=merchant_id,
                captured_amount=amount,
                captured_at=capture_time,
            ),
        )
        self._events.append(payment_captured)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                classification=LabelClassification.LEGITIMATE,
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_captured.envelope.event_id,
                description="LL-01 family payment captured",
            )
        )

        # Order shipped
        ship_time = UTCDateTime(value=capture_time.value + timedelta(hours=random.randint(4, 48)))

        order_shipped = OrderShippedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_SHIPPED,
                occurred_at=ship_time,
                received_at=ship_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderShippedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                shipped_at=ship_time,
            ),
        )
        self._events.append(order_shipped)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                classification=LabelClassification.LEGITIMATE,
                customer_id=customer_id,
                event_id=order_shipped.envelope.event_id,
                description="LL-01 family order shipped",
            )
        )

        # Order delivered
        deliver_time = UTCDateTime(value=ship_time.value + timedelta(days=random.randint(1, 7)))

        order_delivered = OrderDeliveredEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_DELIVERED,
                occurred_at=deliver_time,
                received_at=deliver_time,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderDeliveredPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                delivered_at=deliver_time,
            ),
        )
        self._events.append(order_delivered)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                classification=LabelClassification.LEGITIMATE,
                customer_id=customer_id,
                event_id=order_delivered.envelope.event_id,
                description="LL-01 family order delivered",
            )
        )

        # Natural refund variation - only some orders get refunds
        if random.random() < 0.4:  # 40% refund rate (lower than AS-01)
            refund_id = RefundId.generate()
            refund_paise = random.randint(
                int(amount.amount_paise * 0.2),
                int(amount.amount_paise * 0.7),
            )
            refund_amount = Money.of_paise(refund_paise)

            # Natural timing variation - spread over days/weeks
            refund_request_delay = random.randint(1, 21)  # 1-21 days
            refund_request_time = UTCDateTime(value=deliver_time.value + timedelta(days=refund_request_delay))

            # Varied reason codes (not coordinated)
            reason_codes = [
                RefundReasonCode.DEFECTIVE,
                RefundReasonCode.WRONG_ITEM,
                RefundReasonCode.DAMAGED_IN_TRANSIT,
                RefundReasonCode.CHANGED_MIND,
            ]
            reason_code = random.choice(reason_codes)

            refund_requested = RefundRequestedEvent(
                envelope=EventEnvelope(
                    event_id=self._generate_event_id(),
                    event_type=EventType.REFUND_REQUESTED,
                    occurred_at=refund_request_time,
                    received_at=refund_request_time,
                    source=DataSource.SIMULATOR,
                ),
                payload=RefundRequestedPayload(
                    refund_id=refund_id,
                    payment_id=payment_id,
                    order_id=order_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    amount=refund_amount,
                    reason_code=reason_code,
                ),
            )
            self._events.append(refund_requested)
            self._labels.append(
                GroundTruthLabel(
                    scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                    classification=LabelClassification.LEGITIMATE,
                    customer_id=customer_id,
                    payment_id=payment_id,
                    refund_id=refund_id,
                    event_id=refund_requested.envelope.event_id,
                    description="LL-01 family refund request (natural variation)",
                )
            )

            # Refund created
            created_time = UTCDateTime(value=refund_request_time.value + timedelta(hours=random.randint(1, 24)))

            refund_created = RefundCreatedEvent(
                envelope=EventEnvelope(
                    event_id=self._generate_event_id(),
                    event_type=EventType.REFUND_CREATED,
                    occurred_at=created_time,
                    received_at=created_time,
                    source=DataSource.SIMULATOR,
                ),
                payload=RefundCreatedPayload(
                    refund_id=refund_id,
                    payment_id=payment_id,
                    merchant_id=merchant_id,
                    created_at=created_time,
                ),
            )
            self._events.append(refund_created)
            self._labels.append(
                GroundTruthLabel(
                    scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                    classification=LabelClassification.LEGITIMATE,
                    customer_id=customer_id,
                    payment_id=payment_id,
                    refund_id=refund_id,
                    event_id=refund_created.envelope.event_id,
                    description="LL-01 family refund created",
                )
            )

            # Refund processed
            processed_time = UTCDateTime(value=created_time.value + timedelta(days=random.randint(1, 5)))

            refund_processed = RefundProcessedEvent(
                envelope=EventEnvelope(
                    event_id=self._generate_event_id(),
                    event_type=EventType.REFUND_PROCESSED,
                    occurred_at=processed_time,
                    received_at=processed_time,
                    source=DataSource.SIMULATOR,
                ),
                payload=RefundProcessedPayload(
                    refund_id=refund_id,
                    payment_id=payment_id,
                    merchant_id=merchant_id,
                    processed_at=processed_time,
                    processed_amount=refund_amount,
                ),
            )
            self._events.append(refund_processed)
            self._labels.append(
                GroundTruthLabel(
                    scenario_type=ScenarioType.LL01_LEGITIMATE_FAMILY,
                    classification=LabelClassification.LEGITIMATE,
                    customer_id=customer_id,
                    payment_id=payment_id,
                    refund_id=refund_id,
                    event_id=refund_processed.envelope.event_id,
                    description="LL-01 family refund processed",
                )
            )


# Convenience instances
AS01_DENSE_COORDINATED_REFUND_RING = AS01_DenseCoordinatedRefundRing
LL01_LEGITIMATE_FAMILY = LL01_LegitimateFamily
