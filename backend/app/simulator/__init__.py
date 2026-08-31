"""Refund Sentinel Simulator — Minimal deterministic demo dataset generator.

The simulator generates synthetic financial events for testing and demonstration.
It produces deterministic output given a fixed seed and uses existing domain models
to ensure all generated events validate against the production schema.

Key components:
- Scenario identifiers (AS-01, LL-01, etc.)
- Ground-truth labels for abuse/legitimate classification
- Background population generator
- Abuse scenario generators
- CLI for demo dataset generation

The simulator does NOT bypass validation or persistence. All events are loaded
through the existing IngestionService to ensure they pass the same ingestion
pipeline as production events.
"""

from backend.app.simulator.labels import (
    GroundTruthLabel,
    ScenarioType,
    SimulationOutput,
)
from backend.app.simulator.scenarios import (
    AS01_DENSE_COORDINATED_REFUND_RING,
    LL01_LEGITIMATE_FAMILY,
    Scenario,
)

__all__ = [
    "GroundTruthLabel",
    "ScenarioType",
    "SimulationOutput",
    "AS01_DENSE_COORDINATED_REFUND_RING",
    "LL01_LEGITIMATE_FAMILY",
    "Scenario",
]
