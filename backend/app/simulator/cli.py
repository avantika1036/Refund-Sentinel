"""CLI for demo dataset generation.

Provides a generate-demo command that:
- Uses DEMO_SEED for deterministic output
- Generates background population
- Injects AS-01 and LL-01 scenarios
- Loads events through existing IngestionService
- Stores ground-truth labels
"""

from __future__ import annotations

import sys
from typing import NoReturn

from backend.app.config import settings
from backend.app.persistence.ingestion_service import IngestionService
from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.labels import SimulationOutput
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    LL01_LEGITIMATE_FAMILY,
)


class SimulatorCLI:
    """Command-line interface for simulator operations."""

    @staticmethod
    def generate_demo() -> None:
        """Generate and load the demo dataset.

        Uses DEMO_SEED for deterministic output. Generates:
        - Background population
        - AS-01 dense coordinated refund ring
        - LL-01 legitimate family

        All events are loaded through the existing IngestionService.
        Ground-truth labels are stored separately.
        """
        seed = settings.demo_seed
        print(f"Generating demo dataset with seed: {seed}")

        # Generate background population
        print("Generating background population...")
        background_gen = BackgroundPopulationGenerator(seed=seed)
        background_output = background_gen.generate(
            num_customers=20,
            num_orders_per_customer=3,
            refund_probability=0.15,
        )
        print(f"Generated {len(background_output.events)} background events")

        # Generate AS-01 scenario
        print("Generating AS-01 dense coordinated refund ring...")
        as01_gen = AS01_DENSE_COORDINATED_REFUND_RING(seed=seed + 1000)
        as01_output = as01_gen.generate(
            num_customers=5,
            orders_per_customer=3,
        )
        print(f"Generated {len(as01_output.events)} AS-01 events")

        # Generate LL-01 scenario
        print("Generating LL-01 legitimate family...")
        ll01_gen = LL01_LEGITIMATE_FAMILY(seed=seed + 2000)
        ll01_output = ll01_gen.generate(
            num_family_members=4,
            orders_per_member=2,
        )
        print(f"Generated {len(ll01_output.events)} LL-01 events")

        # Combine all events
        all_events = (
            background_output.events
            + as01_output.events
            + ll01_output.events
        )
        all_labels = (
            background_output.labels
            + as01_output.labels
            + ll01_output.labels
        )

        print(f"Total events to ingest: {len(all_events)}")
        print(f"Total ground-truth labels: {len(all_labels)}")

        # Load events through IngestionService
        print("Loading events through IngestionService...")
        ingestion_service = IngestionService()

        retained_count = 0
        duplicate_count = 0
        pending_count = 0
        quarantined_count = 0

        for event in all_events:
            result = ingestion_service.ingest(event)
            if result.ingestion_outcome.value == "retained":
                retained_count += 1
            elif result.ingestion_outcome.value == "duplicate":
                duplicate_count += 1
            elif result.ingestion_outcome.value == "pending":
                pending_count += 1
            elif result.ingestion_outcome.value == "quarantined":
                quarantined_count += 1

        print(f"Ingestion complete:")
        print(f"  Retained: {retained_count}")
        print(f"  Duplicate: {duplicate_count}")
        print(f"  Pending: {pending_count}")
        print(f"  Quarantined: {quarantined_count}")

        # Store ground-truth labels (in-memory for now)
        # In future phases, this could be persisted to a separate table
        print(f"Ground-truth labels stored in memory: {len(all_labels)}")
        print("\nDemo generation complete.")

        # Print summary
        print("\n--- Summary ---")
        print(f"Background events: {len(background_output.events)}")
        print(f"AS-01 events: {len(as01_output.events)}")
        print(f"LL-01 events: {len(ll01_output.events)}")
        print(f"Total events: {len(all_events)}")
        print(f"Total labels: {len(all_labels)}")
        print(f"Abuse labels: {len([l for l in all_labels if l.is_abuse()])}")
        print(f"Legitimate labels: {len([l for l in all_labels if l.is_legitimate()])}")


def main() -> NoReturn:
    """Entry point for the simulator CLI."""
    try:
        SimulatorCLI.generate_demo()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
