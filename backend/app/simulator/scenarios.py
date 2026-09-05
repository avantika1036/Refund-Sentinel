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
        seed_int = self._rng.getrandbits(128)
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
        amount_paise = self._rng.randint(5000, 30000)  # ₹50 to ₹300
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

        # Coordinated refund request. AS-01 intentionally mixes realized and
        # still-pending abuse so the operational dashboard exercises both
        # deterministic rule evidence and pending financial exposure.
        #
        # order_idx == 0: rapid, full refund before delivery and left REQUESTED.
        # order_idx >= 1: rapid, full refund before delivery and later processed.
        #
        # This keeps the scenario behaviorally coordinated while ensuring that
        # the demo contains visible R01/R02/R03/R06 evidence and genuine
        # pending refund exposure instead of every refund ending PROCESSED.
        refund_id = RefundId.generate()

        # Every AS-01 refund is rapid and full, with a small deterministic
        # offset so the ring remains tightly time-aligned. The first order for
        # each customer stays pending; later orders are processed. This keeps
        # both deterministic and behavioral evidence strong while exercising
        # both realized and pending exposure paths.
        refund_amount = amount
        refund_request_time = UTCDateTime(
            value=capture_time.value
            + timedelta(
                minutes=30 + customer_idx * 5 + order_idx * 10
            )
        )
        leave_pending = order_idx == 0

        # Same reason code reinforces the coordinated-reason signal.
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
                reason_code=RefundReasonCode.DEFECTIVE,
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
                description=(
                    "AS-01 rapid pending refund request (DEFECTIVE)"
                    if leave_pending
                    else "AS-01 coordinated refund request (DEFECTIVE)"
                ),
            )
        )

        # Pending refunds intentionally stop at REQUESTED so they contribute to
        # pending_refund_exposure. All other AS-01 refunds continue through the
        # normal created -> processed lifecycle.
        if leave_pending:
            return

        created_time = UTCDateTime(
            value=refund_request_time.value + timedelta(minutes=30)
        )

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

        processed_time = UTCDateTime(
            value=created_time.value + timedelta(days=1)
        )

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
        amount_paise = self._rng.randint(3000, 40000)  # ₹30 to ₹400
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
        payment_time = UTCDateTime(value=order_time.value + timedelta(minutes=self._rng.randint(5, 45)))

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
        capture_time = UTCDateTime(value=payment_time.value + timedelta(minutes=self._rng.randint(2, 15)))

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
        ship_time = UTCDateTime(value=capture_time.value + timedelta(hours=self._rng.randint(4, 48)))

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
        deliver_time = UTCDateTime(value=ship_time.value + timedelta(days=self._rng.randint(1, 7)))

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
        if self._rng.random() < 0.4:  # 40% refund rate (lower than AS-01)
            refund_id = RefundId.generate()
            refund_paise = self._rng.randint(
                int(amount.amount_paise * 0.2),
                int(amount.amount_paise * 0.7),
            )
            refund_amount = Money.of_paise(refund_paise)

            # Natural timing variation - spread over days/weeks
            refund_request_delay = self._rng.randint(1, 21)  # 1-21 days
            refund_request_time = UTCDateTime(value=deliver_time.value + timedelta(days=refund_request_delay))

            # Varied reason codes (not coordinated)
            reason_codes = [
                RefundReasonCode.DEFECTIVE,
                RefundReasonCode.WRONG_ITEM,
                RefundReasonCode.DAMAGED_IN_TRANSIT,
                RefundReasonCode.CHANGED_MIND,
            ]
            reason_code = self._rng.choice(reason_codes)

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
            created_time = UTCDateTime(value=refund_request_time.value + timedelta(hours=self._rng.randint(1, 24)))

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
            processed_time = UTCDateTime(value=created_time.value + timedelta(days=self._rng.randint(1, 5)))

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


class _ParameterizedLifecycleScenario(Scenario):
    """Reusable lifecycle emitter for additional simulator scenarios.

    The scenario generators below deliberately vary only observable behavior
    and structure. Ground-truth metadata is kept strictly outside the feature
    vector and is used only for supervised training/evaluation.
    """

    def _record(
        self,
        event: AnyDomainEvent,
        *,
        scenario_type: ScenarioType,
        classification: LabelClassification,
        customer_id: CustomerId,
        payment_id: PaymentId | None = None,
        refund_id: RefundId | None = None,
        description: str,
    ) -> None:
        self._events.append(event)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=scenario_type,
                classification=classification,
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=event.envelope.event_id,
                description=description,
            )
        )

    def _emit_lifecycle(
        self,
        *,
        scenario_type: ScenarioType,
        classification: LabelClassification,
        customer_id: CustomerId,
        merchant_id: MerchantId,
        address_id: AddressId,
        device_id: DeviceId,
        order_time: datetime,
        amount_paise: int,
        refund_fraction: float | None,
        refund_delay: timedelta | None,
        reason_code: RefundReasonCode | None,
        shipping_delay: timedelta,
        delivery_delay: timedelta,
        description_prefix: str,
    ) -> None:
        """Emit one complete observable commerce lifecycle."""
        order_at = UTCDateTime(value=order_time)
        order_id = OrderId.generate()
        amount = Money.of_paise(amount_paise)

        order_created = OrderCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_CREATED,
                occurred_at=order_at,
                received_at=order_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderCreatedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                shipping_address_id=address_id,
            ),
        )
        self._record(order_created, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, description=f"{description_prefix} order created")

        payment_id = PaymentId.generate()
        payment_at = UTCDateTime(value=order_at.value + timedelta(minutes=self._rng.randint(2, 30)))
        payment_created = PaymentCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CREATED,
                occurred_at=payment_at,
                received_at=payment_at,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCreatedPayload(
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                device_id=device_id,
            ),
        )
        self._record(payment_created, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, payment_id=payment_id,
                     description=f"{description_prefix} payment created")

        captured_at = UTCDateTime(value=payment_at.value + timedelta(minutes=self._rng.randint(1, 12)))
        payment_captured = PaymentCapturedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CAPTURED,
                occurred_at=captured_at,
                received_at=captured_at,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCapturedPayload(
                payment_id=payment_id,
                merchant_id=merchant_id,
                captured_amount=amount,
                captured_at=captured_at,
            ),
        )
        self._record(payment_captured, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, payment_id=payment_id,
                     description=f"{description_prefix} payment captured")

        shipped_at = UTCDateTime(value=captured_at.value + shipping_delay)
        order_shipped = OrderShippedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_SHIPPED,
                occurred_at=shipped_at,
                received_at=shipped_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderShippedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                shipped_at=shipped_at,
            ),
        )
        self._record(order_shipped, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, description=f"{description_prefix} order shipped")

        delivered_at = UTCDateTime(value=shipped_at.value + delivery_delay)
        order_delivered = OrderDeliveredEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_DELIVERED,
                occurred_at=delivered_at,
                received_at=delivered_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderDeliveredPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                delivered_at=delivered_at,
            ),
        )
        self._record(order_delivered, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, description=f"{description_prefix} order delivered")

        if refund_fraction is None or refund_delay is None or reason_code is None:
            return

        refund_id = RefundId.generate()
        refund_amount = Money.of_paise(max(1, int(amount.amount_paise * refund_fraction)))
        requested_at = UTCDateTime(value=delivered_at.value + refund_delay)
        refund_requested = RefundRequestedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_REQUESTED,
                occurred_at=requested_at,
                received_at=requested_at,
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
        self._record(refund_requested, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, payment_id=payment_id, refund_id=refund_id,
                     description=f"{description_prefix} refund requested")

        created_at = UTCDateTime(value=requested_at.value + timedelta(minutes=self._rng.randint(10, 180)))
        refund_created = RefundCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_CREATED,
                occurred_at=created_at,
                received_at=created_at,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundCreatedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                created_at=created_at,
            ),
        )
        self._record(refund_created, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, payment_id=payment_id, refund_id=refund_id,
                     description=f"{description_prefix} refund created")

        processed_at = UTCDateTime(value=created_at.value + timedelta(hours=self._rng.randint(6, 72)))
        refund_processed = RefundProcessedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_PROCESSED,
                occurred_at=processed_at,
                received_at=processed_at,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundProcessedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                processed_at=processed_at,
                processed_amount=refund_amount,
            ),
        )
        self._record(refund_processed, scenario_type=scenario_type, classification=classification,
                     customer_id=customer_id, payment_id=payment_id, refund_id=refund_id,
                     description=f"{description_prefix} refund processed")


class AS02_VelocityRefundAbuse(_ParameterizedLifecycleScenario):
    """AS-02: repeated rapid refunds by the same customers."""

    def generate(
        self,
        num_customers: int = 4,
        refunds_per_customer: int = 6,
    ) -> SimulationOutput:
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 4, 1, tzinfo=timezone.utc)
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            address_id = AddressId.generate()
            device_id = DeviceId.generate()
            for refund_idx in range(refunds_per_customer):
                self._emit_lifecycle(
                    scenario_type=ScenarioType.AS02_VELOCITY_REFUND_ABUSE,
                    classification=LabelClassification.ABUSE,
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    address_id=address_id,
                    device_id=device_id,
                    order_time=base_time + timedelta(hours=customer_idx * 8 + refund_idx * 5),
                    amount_paise=self._rng.randint(12_000, 42_000),
                    refund_fraction=self._rng.uniform(0.9, 1.0),
                    refund_delay=timedelta(hours=self._rng.randint(1, 6)),
                    reason_code=RefundReasonCode.DEFECTIVE,
                    shipping_delay=timedelta(hours=self._rng.randint(1, 4)),
                    delivery_delay=timedelta(hours=self._rng.randint(3, 10)),
                    description_prefix="AS-02 rapid refund velocity",
                )
        return SimulationOutput(events=self._events, labels=self._labels, seed=self._seed)


class AS03_SharedPaymentDeviceRing(_ParameterizedLifecycleScenario):
    """AS-03: customers repeatedly reuse shared structural attributes."""

    def generate(
        self,
        num_customers: int = 6,
        orders_per_customer: int = 3,
    ) -> SimulationOutput:
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 5, 1, tzinfo=timezone.utc)
        shared_device_id = DeviceId.generate()
        shared_address_id = AddressId.generate()
        reasons = [RefundReasonCode.DEFECTIVE, RefundReasonCode.WRONG_ITEM]
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            for order_idx in range(orders_per_customer):
                self._emit_lifecycle(
                    scenario_type=ScenarioType.AS03_SHARED_PAYMENT_DEVICE_RING,
                    classification=LabelClassification.ABUSE,
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    address_id=shared_address_id,
                    device_id=shared_device_id,
                    order_time=base_time + timedelta(hours=customer_idx * 4 + order_idx * 3),
                    amount_paise=self._rng.randint(8_000, 35_000),
                    refund_fraction=self._rng.uniform(0.75, 1.0),
                    refund_delay=timedelta(hours=self._rng.randint(2, 30)),
                    reason_code=self._rng.choice(reasons),
                    shipping_delay=timedelta(hours=self._rng.randint(2, 8)),
                    delivery_delay=timedelta(hours=self._rng.randint(8, 24)),
                    description_prefix="AS-03 shared-attribute ring",
                )
        return SimulationOutput(events=self._events, labels=self._labels, seed=self._seed)


class AS04_IsolatedRefundChurn(_ParameterizedLifecycleScenario):
    """AS-04: high-velocity single-account refund churn without shared topology.

    This held-out abuse family is intentionally structurally isolated. It tests
    whether individual behavior and financial/temporal signals can detect abuse
    that a graph-only baseline cannot observe.
    """

    def generate(
        self,
        num_customers: int = 3,
        refunds_per_customer: int = 7,
    ) -> SimulationOutput:
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            for refund_idx in range(refunds_per_customer):
                self._emit_lifecycle(
                    scenario_type=ScenarioType.AS04_ISOLATED_REFUND_CHURN,
                    classification=LabelClassification.ABUSE,
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    # Deliberately unique identifiers keep each customer's
                    # refund behavior structurally isolated from other accounts.
                    address_id=AddressId.generate(),
                    device_id=DeviceId.generate(),
                    order_time=base_time + timedelta(
                        days=customer_idx * 3,
                        hours=refund_idx * 4,
                    ),
                    amount_paise=self._rng.randint(18_000, 48_000),
                    refund_fraction=self._rng.uniform(0.95, 1.0),
                    refund_delay=timedelta(hours=self._rng.randint(1, 4)),
                    reason_code=RefundReasonCode.DEFECTIVE,
                    shipping_delay=timedelta(hours=self._rng.randint(1, 3)),
                    delivery_delay=timedelta(hours=self._rng.randint(3, 8)),
                    description_prefix="AS-04 isolated refund churn",
                )
        return SimulationOutput(events=self._events, labels=self._labels, seed=self._seed)


class LL03_SharedHousehold(_ParameterizedLifecycleScenario):
    """LL-03: legitimate household sharing device and delivery address.

    This held-out negative control intentionally resembles a large structural
    cluster while keeping refund behavior sparse and naturally distributed. It
    prevents
    a graph-only baseline from appearing perfect merely because every connected
    component is labelled abusive.
    """

    def generate(
        self,
        num_customers: int = 6,
        orders_per_customer: int = 7,
    ) -> SimulationOutput:
        merchant_id = MerchantId.generate()
        base_time = datetime(2023, 5, 1, tzinfo=timezone.utc)
        shared_device_id = DeviceId.generate()
        shared_address_id = AddressId.generate()
        reasons = [
            RefundReasonCode.DEFECTIVE,
            RefundReasonCode.WRONG_ITEM,
            RefundReasonCode.DAMAGED_IN_TRANSIT,
        ]
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            for order_idx in range(orders_per_customer):
                should_refund = order_idx == 5
                self._emit_lifecycle(
                    scenario_type=ScenarioType.LL03_SHARED_HOUSEHOLD,
                    classification=LabelClassification.LEGITIMATE,
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    address_id=shared_address_id,
                    device_id=shared_device_id,
                    order_time=base_time + timedelta(
                        days=customer_idx * 5 + order_idx * 9
                    ),
                    amount_paise=self._rng.randint(4_000, 40_000),
                    refund_fraction=(
                        self._rng.uniform(0.25, 0.6) if should_refund else None
                    ),
                    refund_delay=(
                        timedelta(days=self._rng.randint(8, 24))
                        if should_refund else None
                    ),
                    reason_code=self._rng.choice(reasons) if should_refund else None,
                    shipping_delay=timedelta(hours=self._rng.randint(12, 36)),
                    delivery_delay=timedelta(days=self._rng.randint(2, 7)),
                    description_prefix="LL-03 shared legitimate household",
                )
        return SimulationOutput(events=self._events, labels=self._labels, seed=self._seed)


class LL02_FrequentShopper(_ParameterizedLifecycleScenario):
    """LL-02: frequent legitimate shoppers with substantial normal history."""

    def generate(
        self,
        num_customers: int = 5,
        orders_per_customer: int = 12,
    ) -> SimulationOutput:
        merchant_id = MerchantId.generate()
        base_time = datetime(2023, 8, 1, tzinfo=timezone.utc)
        reasons = [
            RefundReasonCode.DEFECTIVE,
            RefundReasonCode.WRONG_ITEM,
            RefundReasonCode.DAMAGED_IN_TRANSIT,
            RefundReasonCode.CHANGED_MIND,
        ]
        for customer_idx in range(num_customers):
            customer_id = CustomerId.generate()
            address_id = AddressId.generate()
            device_id = DeviceId.generate()
            for order_idx in range(orders_per_customer):
                should_refund = order_idx in {4, 10}
                self._emit_lifecycle(
                    scenario_type=ScenarioType.LL02_FREQUENT_SHOPPER,
                    classification=LabelClassification.LEGITIMATE,
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    address_id=address_id,
                    device_id=device_id,
                    order_time=base_time + timedelta(days=customer_idx * 2 + order_idx * 8),
                    amount_paise=self._rng.randint(3_000, 55_000),
                    refund_fraction=(self._rng.uniform(0.25, 0.7) if should_refund else None),
                    refund_delay=(timedelta(days=self._rng.randint(5, 28)) if should_refund else None),
                    reason_code=(self._rng.choice(reasons) if should_refund else None),
                    shipping_delay=timedelta(hours=self._rng.randint(8, 48)),
                    delivery_delay=timedelta(days=self._rng.randint(1, 6)),
                    description_prefix="LL-02 frequent legitimate shopper",
                )
        return SimulationOutput(events=self._events, labels=self._labels, seed=self._seed)


# Convenience instances
AS01_DENSE_COORDINATED_REFUND_RING = AS01_DenseCoordinatedRefundRing
AS02_VELOCITY_REFUND_ABUSE = AS02_VelocityRefundAbuse
AS03_SHARED_PAYMENT_DEVICE_RING = AS03_SharedPaymentDeviceRing
AS04_ISOLATED_REFUND_CHURN = AS04_IsolatedRefundChurn
LL01_LEGITIMATE_FAMILY = LL01_LegitimateFamily
LL02_FREQUENT_SHOPPER = LL02_FrequentShopper
LL03_SHARED_HOUSEHOLD = LL03_SharedHousehold
