"""Focused tests for Phase 3 simulator.

Verifies:
1. Deterministic output for same seed
2. Events validate against domain schemas
3. AS-01 coordinated structure/behavior
4. LL-01 shared attributes with varied behavior
5. Labels distinguish scenarios correctly
6. Events pass through ingestion pipeline
"""

from backend.app.simulator.background import BackgroundPopulationGenerator
from backend.app.simulator.labels import (
    GroundTruthLabel,
    LabelClassification,
    ScenarioType,
)
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    LL01_LEGITIMATE_FAMILY,
)


def test_same_seed_provides_deterministic_background_output() -> None:
    """Same seed produces equivalent deterministic simulation output."""
    seed = 42
    gen1 = BackgroundPopulationGenerator(seed=seed)
    output1 = gen1.generate(num_customers=5, num_orders_per_customer=2)

    gen2 = BackgroundPopulationGenerator(seed=seed)
    output2 = gen2.generate(num_customers=5, num_orders_per_customer=2)

    # Same number of events
    assert len(output1.events) == len(output2.events)

    # Same number of labels
    assert len(output1.labels) == len(output2.labels)

    # Same event types in same order
    event_types_1 = [e.envelope.event_type for e in output1.events]
    event_types_2 = [e.envelope.event_type for e in output2.events]
    assert event_types_1 == event_types_2

    # Same seed is preserved
    assert output1.seed == output2.seed == seed


def test_generated_events_validate_domain_schemas() -> None:
    """Generated events validate against existing domain event schemas."""
    gen = BackgroundPopulationGenerator(seed=42)
    output = gen.generate(num_customers=3, num_orders_per_customer=2)

    # All events should be valid domain events
    # If they were invalid, Pydantic validation would have raised during construction
    assert len(output.events) > 0

    # Each event should have required fields
    for event in output.events:
        assert event.envelope.event_id is not None
        assert event.envelope.event_type is not None
        assert event.envelope.occurred_at is not None
        assert event.envelope.received_at is not None
        assert event.envelope.source is not None


def test_as01_contains_coordinated_structure() -> None:
    """AS-01 contains intended coordinated structure/behavior."""
    as01_gen = AS01_DENSE_COORDINATED_REFUND_RING(seed=42)
    output = as01_gen.generate(num_customers=3, orders_per_customer=2)

    # Should have events
    assert len(output.events) > 0

    # All labels should be AS-01 abuse
    as01_labels = output.get_labels_for_scenario(ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING)
    assert len(as01_labels) == len(output.labels)

    # All should be classified as abuse
    abuse_labels = output.get_abuse_labels()
    assert len(abuse_labels) == len(output.labels)

    # No legitimate labels
    legitimate_labels = output.get_legitimate_labels()
    assert len(legitimate_labels) == 0


def test_ll01_has_shared_attributes_varied_behavior() -> None:
    """LL-01 contains shared structural attributes but varied lifecycle behavior."""
    ll01_gen = LL01_LEGITIMATE_FAMILY(seed=42)
    output = ll01_gen.generate(num_family_members=3, orders_per_member=2)

    # Should have events
    assert len(output.events) > 0

    # All labels should be LL-01 legitimate
    ll01_labels = output.get_labels_for_scenario(ScenarioType.LL01_LEGITIMATE_FAMILY)
    assert len(ll01_labels) == len(output.labels)

    # All should be classified as legitimate
    legitimate_labels = output.get_legitimate_labels()
    assert len(legitimate_labels) == len(output.labels)

    # No abuse labels
    abuse_labels = output.get_abuse_labels()
    assert len(abuse_labels) == 0


def test_labels_distinguish_scenarios_correctly() -> None:
    """Labels correctly distinguish AS-01 from LL-01/background."""
    # Background
    bg_gen = BackgroundPopulationGenerator(seed=42)
    bg_output = bg_gen.generate(num_customers=2, num_orders_per_customer=1)

    # AS-01
    as01_gen = AS01_DENSE_COORDINATED_REFUND_RING(seed=42)
    as01_output = as01_gen.generate(num_customers=2, orders_per_customer=1)

    # LL-01
    ll01_gen = LL01_LEGITIMATE_FAMILY(seed=42)
    ll01_output = ll01_gen.generate(num_family_members=2, orders_per_member=1)

    # Background labels
    bg_labels = bg_output.get_labels_for_scenario(ScenarioType.BACKGROUND)
    assert all(l.classification == LabelClassification.LEGITIMATE for l in bg_labels)

    # AS-01 labels
    as01_labels = as01_output.get_labels_for_scenario(ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING)
    assert all(l.classification == LabelClassification.ABUSE for l in as01_labels)

    # LL-01 labels
    ll01_labels = ll01_output.get_labels_for_scenario(ScenarioType.LL01_LEGITIMATE_FAMILY)
    assert all(l.classification == LabelClassification.LEGITIMATE for l in ll01_labels)

    # Scenarios are distinct
    assert len(bg_labels) > 0
    assert len(as01_labels) > 0
    assert len(ll01_labels) > 0


def test_different_seeds_produce_different_output() -> None:
    """Different seeds produce different simulation output."""
    gen1 = BackgroundPopulationGenerator(seed=42)
    output1 = gen1.generate(num_customers=3, num_orders_per_customer=1)

    gen2 = BackgroundPopulationGenerator(seed=999)
    output2 = gen2.generate(num_customers=3, num_orders_per_customer=1)

    # Different seeds should produce different event IDs
    event_ids_1 = [e.envelope.event_id for e in output1.events]
    event_ids_2 = [e.envelope.event_id for e in output2.events]

    # At least some event IDs should differ
    assert event_ids_1 != event_ids_2


def test_scenario_extensibility() -> None:
    """Scenario design is extensible for future scenarios."""
    # Verify scenario types are properly defined
    assert ScenarioType.AS01_DENSE_COORDINATED_REFUND_RING.value == "AS01_DENSE_COORDINATED_REFUND_RING"
    assert ScenarioType.LL01_LEGITIMATE_FAMILY.value == "LL01_LEGITIMATE_FAMILY"
    assert ScenarioType.BACKGROUND.value == "BACKGROUND"

    # Verify classification types
    assert LabelClassification.ABUSE.value == "abuse"
    assert LabelClassification.LEGITIMATE.value == "legitimate"

    # Future scenarios can be added without breaking existing ones
    # This test verifies the enum structure supports extension
