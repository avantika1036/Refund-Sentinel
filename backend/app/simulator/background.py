"""Deterministic background population generator.

Generates a legitimate customer population with normal payment/order/refund
lifecycle events. Uses explicit seed for reproducible output.
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
from backend.app.domain.identifiers import CustomerId, EventId, MerchantId, OrderId, PaymentId, RefundId
from backend.app.domain.value_objects import Money, UTCDateTime
from backend.app.simulator.labels import GroundTruthLabel, ScenarioType, SimulationOutput

if TYPE_CHECKING:
    pass


class BackgroundPopulationGenerator:
    """Deterministic generator for legitimate background population.

    Uses a fixed seed to produce reproducible output. Generates valid domain
    entities/events using existing domain models. Avoids creating coordinated
    abuse patterns intentionally.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed or settings.demo_seed
        self._rng = random.Random(self._seed)
        self._events: list[AnyDomainEvent] = []
        self._labels: list[GroundTruthLabel] = []

    def generate(
        self,
        num_customers: int = 20,
        num_orders_per_customer: int = 3,
        refund_probability: float = 0.15,
    ) -> SimulationOutput:
        """Generate a background population with normal lifecycle events.

        Args:
            num_customers: Number of distinct customers to generate
            num_orders_per_customer: Average orders per customer
            refund_probability: Probability of a payment being refunded
        """
        merchant_id = MerchantId.generate()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        for customer_idx in range(num_customers):
            customer_id = self._generate_customer_id(customer_idx)
            num_orders = self._rng.randint(1, num_orders_per_customer + 1)

            for order_idx in range(num_orders):
                self._generate_customer_lifecycle(
                    customer_id,
                    merchant_id,
                    base_time,
                    order_idx,
                    refund_probability,
                )

        return SimulationOutput(
            events=self._events,
            labels=self._labels,
            seed=self._seed,
        )

    def _generate_customer_id(self, index: int) -> CustomerId:
        """Generate a deterministic customer ID from index and seed."""
        # Use seed + index to generate deterministic UUID
        seed_value = self._seed + index
        self._rng = random.Random(seed_value)
        return CustomerId.generate()

    def _generate_customer_lifecycle(
        self,
        customer_id: CustomerId,
        merchant_id: MerchantId,
        base_time: datetime,
        order_idx: int,
        refund_probability: float,
    ) -> None:
        """Generate a complete order/payment lifecycle for one customer."""
        # Add random time offset for each order
        time_offset_days = self._rng.randint(0, 30)
        order_time = base_time + timedelta(days=time_offset_days)
        order_time = UTCDateTime(value=order_time)

        # Generate order
        order_id = OrderId.generate()
        amount_paise = self._rng.randint(1000, 50000)  # ₹10 to ₹500
        amount = Money.of_paise(amount_paise)

        order_created = self._create_order_created(
            order_id, merchant_id, customer_id, amount, order_time
        )
        self._events.append(order_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                event_id=order_created.envelope.event_id,
                description="Background order created",
            )
        )

        # Generate payment
        payment_id = PaymentId.generate()
        payment_time = UTCDateTime(value=order_time.value + timedelta(minutes=self._rng.randint(5, 30)))

        payment_created = self._create_payment_created(
            payment_id, order_id, merchant_id, customer_id, amount, payment_time
        )
        self._events.append(payment_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_created.envelope.event_id,
                description="Background payment created",
            )
        )

        # Capture payment
        capture_time = UTCDateTime(value=payment_time.value + timedelta(minutes=self._rng.randint(1, 10)))

        payment_captured = self._create_payment_captured(
            payment_id, merchant_id, amount, capture_time
        )
        self._events.append(payment_captured)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                payment_id=payment_id,
                event_id=payment_captured.envelope.event_id,
                description="Background payment captured",
            )
        )

        # Ship order
        ship_time = UTCDateTime(value=capture_time.value + timedelta(hours=self._rng.randint(1, 24)))

        order_shipped = self._create_order_shipped(order_id, merchant_id, ship_time)
        self._events.append(order_shipped)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                event_id=order_shipped.envelope.event_id,
                description="Background order shipped",
            )
        )

        # Deliver order
        deliver_time = UTCDateTime(value=ship_time.value + timedelta(days=self._rng.randint(1, 5)))

        order_delivered = self._create_order_delivered(order_id, merchant_id, deliver_time)
        self._events.append(order_delivered)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                event_id=order_delivered.envelope.event_id,
                description="Background order delivered",
            )
        )

        # Possibly generate refund
        if self._rng.random() < refund_probability:
            self._generate_refund_lifecycle(
                customer_id,
                payment_id,
                order_id,
                merchant_id,
                amount,
                deliver_time,
            )

    def _generate_refund_lifecycle(
        self,
        customer_id: CustomerId,
        payment_id: PaymentId,
        order_id: OrderId,
        merchant_id: MerchantId,
        original_amount: Money,
        deliver_time: UTCDateTime,
    ) -> None:
        """Generate a refund lifecycle with natural variation."""
        refund_id = RefundId.generate()

        # Refund amount (partial refunds are common)
        refund_paise = self._rng.randint(
            int(original_amount.amount_paise * 0.1),
            int(original_amount.amount_paise * 0.9),
        )
        refund_amount = Money.of_paise(refund_paise)

        # Refund request timing (natural variation)
        request_delay_days = self._rng.randint(1, 14)
        request_time = UTCDateTime(value=deliver_time.value + timedelta(days=request_delay_days))

        # Refund reason (natural variation)
        reason_codes = list(RefundReasonCode)
        reason_code = self._rng.choice(reason_codes)

        refund_requested = self._create_refund_requested(
            refund_id,
            payment_id,
            order_id,
            merchant_id,
            customer_id,
            refund_amount,
            request_time,
            reason_code,
        )
        self._events.append(refund_requested)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_requested.envelope.event_id,
                description="Background refund requested",
            )
        )

        # Refund created
        created_time = UTCDateTime(value=request_time.value + timedelta(minutes=self._rng.randint(5, 60)))

        refund_created = self._create_refund_created(
            refund_id, payment_id, merchant_id, created_time
        )
        self._events.append(refund_created)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_created.envelope.event_id,
                description="Background refund created",
            )
        )

        # Refund processed
        processed_time = UTCDateTime(value=created_time.value + timedelta(days=self._rng.randint(1, 7)))

        refund_processed = self._create_refund_processed(
            refund_id, payment_id, merchant_id, refund_amount, processed_time
        )
        self._events.append(refund_processed)
        self._labels.append(
            GroundTruthLabel(
                scenario_type=ScenarioType.BACKGROUND,
                classification="legitimate",  # type: ignore
                customer_id=customer_id,
                payment_id=payment_id,
                refund_id=refund_id,
                event_id=refund_processed.envelope.event_id,
                description="Background refund processed",
            )
        )

    def _create_order_created(
        self,
        order_id: OrderId,
        merchant_id: MerchantId,
        customer_id: CustomerId,
        amount: Money,
        occurred_at: UTCDateTime,
    ) -> OrderCreatedEvent:
        return OrderCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_CREATED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderCreatedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
            ),
        )

    def _create_payment_created(
        self,
        payment_id: PaymentId,
        order_id: OrderId,
        merchant_id: MerchantId,
        customer_id: CustomerId,
        amount: Money,
        occurred_at: UTCDateTime,
    ) -> PaymentCreatedEvent:
        return PaymentCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CREATED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCreatedPayload(
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
            ),
        )

    def _create_payment_captured(
        self,
        payment_id: PaymentId,
        merchant_id: MerchantId,
        amount: Money,
        occurred_at: UTCDateTime,
    ) -> PaymentCapturedEvent:
        return PaymentCapturedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.PAYMENT_CAPTURED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=PaymentCapturedPayload(
                payment_id=payment_id,
                merchant_id=merchant_id,
                captured_amount=amount,
                captured_at=occurred_at,
            ),
        )

    def _create_order_shipped(
        self,
        order_id: OrderId,
        merchant_id: MerchantId,
        occurred_at: UTCDateTime,
    ) -> OrderShippedEvent:
        return OrderShippedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_SHIPPED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderShippedPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                shipped_at=occurred_at,
            ),
        )

    def _create_order_delivered(
        self,
        order_id: OrderId,
        merchant_id: MerchantId,
        occurred_at: UTCDateTime,
    ) -> OrderDeliveredEvent:
        return OrderDeliveredEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.ORDER_DELIVERED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=OrderDeliveredPayload(
                order_id=order_id,
                merchant_id=merchant_id,
                delivered_at=occurred_at,
            ),
        )

    def _create_refund_requested(
        self,
        refund_id: RefundId,
        payment_id: PaymentId,
        order_id: OrderId,
        merchant_id: MerchantId,
        customer_id: CustomerId,
        amount: Money,
        occurred_at: UTCDateTime,
        reason_code: RefundReasonCode,
    ) -> RefundRequestedEvent:
        return RefundRequestedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_REQUESTED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundRequestedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                order_id=order_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                amount=amount,
                reason_code=reason_code,
            ),
        )

    def _create_refund_created(
        self,
        refund_id: RefundId,
        payment_id: PaymentId,
        merchant_id: MerchantId,
        occurred_at: UTCDateTime,
    ) -> RefundCreatedEvent:
        return RefundCreatedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_CREATED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundCreatedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                created_at=occurred_at,
            ),
        )

    def _create_refund_processed(
        self,
        refund_id: RefundId,
        payment_id: PaymentId,
        merchant_id: MerchantId,
        amount: Money,
        occurred_at: UTCDateTime,
    ) -> RefundProcessedEvent:
        return RefundProcessedEvent(
            envelope=EventEnvelope(
                event_id=self._generate_event_id(),
                event_type=EventType.REFUND_PROCESSED,
                occurred_at=occurred_at,
                received_at=occurred_at,
                source=DataSource.SIMULATOR,
            ),
            payload=RefundProcessedPayload(
                refund_id=refund_id,
                payment_id=payment_id,
                merchant_id=merchant_id,
                processed_at=occurred_at,
                processed_amount=amount,
            ),
        )

    def _generate_event_id(self) -> EventId:
        """Generate a deterministic event ID."""
        import uuid
        # Use random int to seed UUID for determinism
        seed_int = self._rng.getrandbits(128)
        return EventId(uuid.UUID(int=seed_int))
