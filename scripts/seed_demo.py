"""Seed the persistent database with the canonical Refund Sentinel demo.

The demo generator uses distinct deterministic seeds per scenario family. This
is important because simulator event IDs are deterministic within a seed; using
the same seed for several generators can otherwise create event-ID collisions
in the durable event ledger and silently omit complete scenario families.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import delete

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.persistence.database import SessionLocal, engine
from backend.app.persistence.models import (
    Base,
    EventModel,
    IngestionRecordModel,
    PendingEventModel,
    QuarantineRecordModel,
)
from backend.app.persistence.repositories.events import EventRepository
from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    AS02_VELOCITY_REFUND_ABUSE,
    AS03_SHARED_PAYMENT_DEVICE_RING,
    LL01_LEGITIMATE_FAMILY,
    LL02_FREQUENT_SHOPPER,
)


def _scenario_seed(base_seed: int, offset: int) -> int:
    return base_seed + offset


def _reset_demo_ledger(session) -> None:
    """Clear durable event-ledger state for a reproducible demo dataset."""
    session.execute(delete(IngestionRecordModel))
    session.execute(delete(PendingEventModel))
    session.execute(delete(QuarantineRecordModel))
    session.execute(delete(EventModel))
    session.commit()


def seed_demo_database(
    num_background_customers: int = 25,
    seed: int = 42,
    *,
    reset_database: bool = True,
) -> None:
    print("================================================================")
    print(" REFUND SENTINEL: SEEDING DEMO SCENARIOS INTO DATABASE")
    print("================================================================")

    print("\n[1/3] Ensuring persistence schema tables exist...")
    Base.metadata.create_all(bind=engine)

    if reset_database:
        print("  - Resetting existing event-ledger data for a clean demo...")
        with SessionLocal() as session:
            _reset_demo_ledger(session)

    print("\n[2/3] Generating synthetic demo events...")
    all_events = []

    generators = [
        (
            f"Generating {num_background_customers} background customers...",
            lambda: BackgroundPopulationGenerator(
                seed=_scenario_seed(seed, 0)
            ).generate(num_customers=num_background_customers),
        ),
        (
            "Generating AS-01: Dense Coordinated Sybil Ring...",
            lambda: AS01_DENSE_COORDINATED_REFUND_RING(
                seed=_scenario_seed(seed, 101)
            ).generate(),
        ),
        (
            "Generating AS-02: Rapid Velocity Abuse...",
            lambda: AS02_VELOCITY_REFUND_ABUSE(
                seed=_scenario_seed(seed, 202)
            ).generate(),
        ),
        (
            "Generating AS-03: Shared Card & Device Ring...",
            lambda: AS03_SHARED_PAYMENT_DEVICE_RING(
                seed=_scenario_seed(seed, 303)
            ).generate(),
        ),
        (
            "Generating LL-01: Legitimate Family Group (Lookalike Negative Control)...",
            lambda: LL01_LEGITIMATE_FAMILY(
                seed=_scenario_seed(seed, 404)
            ).generate(),
        ),
        (
            "Generating LL-02: Frequent Shopper (Lookalike Negative Control)...",
            lambda: LL02_FREQUENT_SHOPPER(
                seed=_scenario_seed(seed, 505)
            ).generate(),
        ),
    ]

    for message, generator in generators:
        print(f"  - {message}")
        all_events.extend(generator().events)

    all_events.sort(key=lambda event: event.envelope.occurred_at.value)
    expected_count = len(all_events)
    print(f"\nTotal synthetic events to persist: {expected_count}")

    print("\n[3/3] Ingesting domain events into event repository...")
    saved_count = duplicate_count = conflict_count = 0
    with SessionLocal() as session:
        event_repo = EventRepository(session)
        for event in all_events:
            result = event_repo.save(event)
            outcome = result.outcome.value
            if outcome == "inserted":
                saved_count += 1
            elif outcome == "duplicate":
                duplicate_count += 1
            else:
                conflict_count += 1
        session.commit()

    print(f"\nInserted:   {saved_count}")
    print(f"Duplicates: {duplicate_count}")
    print(f"Conflicts:  {conflict_count}")

    if reset_database and (
        saved_count != expected_count
        or duplicate_count != 0
        or conflict_count != 0
    ):
        raise RuntimeError(
            "Demo seed verification failed: a clean database must persist every "
            "generated event exactly once."
        )

    print("Demo database is ready. Restart the backend and refresh the Investigation Queue.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed database with the canonical Refund Sentinel demo scenarios."
    )
    parser.add_argument("--background-customers", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not clear existing event-ledger data before seeding.",
    )
    args = parser.parse_args()

    seed_demo_database(
        num_background_customers=args.background_customers,
        seed=args.seed,
        reset_database=not args.append,
    )


if __name__ == "__main__":
    main()
