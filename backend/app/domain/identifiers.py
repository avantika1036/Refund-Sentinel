"""
Strongly typed identifier wrappers for Refund Sentinel domain objects.

Each entity type has its own ID class. This prevents accidental substitution
of one identifier type for another (e.g. passing a PaymentId where a RefundId
is expected).

Pydantic v2 integration
-----------------------
__get_pydantic_core_schema__ registers a custom validation schema so that
Pydantic enforces the exact concrete subclass at field validation time.

Validation accepts, in order:
  1. An instance of the EXACT subclass (type(value) is cls).
  2. A uuid.UUID — wrapped in the correct subclass.
  3. A str — parsed as UUID then wrapped.
  4. Anything else, including a sibling ID class — rejected with ValueError.

This means a field typed `payment_id: PaymentId` will reject a `RefundId(...)`
at Pydantic validation time, not silently coerce it.

Identity and equality
---------------------
- IDs are immutable after construction.
- Two IDs are equal only if they share the same concrete class AND the same UUID.
- Hash includes the class name so sibling classes with the same UUID hash
  differently and can coexist in the same set or dict without collision.

Serialization
-------------
str(id)           → lowercase hyphenated UUID string.
model_dump() / JSON → str, via the serializer registered in the schema.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class _BaseId:
    """
    Abstract base for all typed identifiers.

    Do not instantiate _BaseId directly. Use a concrete subclass.
    """

    __slots__ = ("_value",)

    def __init__(self, value: uuid.UUID) -> None:
        if not isinstance(value, uuid.UUID):
            raise TypeError(
                f"{self.__class__.__name__} requires a uuid.UUID instance, "
                f"got {type(value).__name__!r}. "
                f"Use {self.__class__.__name__}.from_str() to parse a UUID string."
            )
        object.__setattr__(self, "_value", value)

    # ------------------------------------------------------------------
    # Immutability
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{self.__class__.__name__} is immutable; cannot set {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{self.__class__.__name__} is immutable; cannot delete {name!r}"
        )

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    @property
    def value(self) -> uuid.UUID:
        return object.__getattribute__(self, "_value")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> "_BaseId":
        """Create a new random identifier of this exact subclass."""
        return cls(uuid.uuid4())

    @classmethod
    def from_str(cls, raw: str) -> "_BaseId":
        """
        Parse an identifier from a UUID string.

        Raises ValueError if the string is not a valid UUID format.
        """
        try:
            return cls(uuid.UUID(raw))
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Cannot parse {cls.__name__} from {raw!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Pydantic v2 integration
    # ------------------------------------------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Register a Pydantic v2 validation schema for this ID class.

        The validator function performs an exact type check (type(value) is cls)
        for ID instances, so a RefundId is rejected where a PaymentId is expected
        even though both are subclasses of _BaseId.

        Serialization converts the ID to its string representation for
        model_dump() and JSON output.
        """

        def validate(value: Any) -> "_BaseId":
            if type(value) is cls:
                # Exact type match — already the correct ID class.
                return value
            if isinstance(value, uuid.UUID):
                return cls(value)
            if isinstance(value, str):
                return cls.from_str(value)
            raise ValueError(
                f"Cannot construct {cls.__name__} from {type(value).__name__!r}. "
                f"Expected a {cls.__name__} instance, a uuid.UUID, or a UUID string. "
                f"A sibling ID type (e.g. RefundId where PaymentId is expected) "
                f"is not accepted."
            )

        return core_schema.with_info_plain_validator_function(
            lambda value, info: validate(value),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            # Different concrete classes are never equal, even with the same UUID.
            return NotImplemented
        return self.value == other.value  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        # Include class name so PaymentId(x) and RefundId(x) hash differently.
        return hash((self.__class__.__name__, self.value))

    def __lt__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return str(self.value) < str(other.value)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"


# ---------------------------------------------------------------------------
# Concrete identifier types — one class per entity type.
# ---------------------------------------------------------------------------


class MerchantId(_BaseId):
    """Identifies a Merchant."""


class CustomerId(_BaseId):
    """Identifies a Customer."""


class OrderId(_BaseId):
    """Identifies an Order."""


class PaymentId(_BaseId):
    """Identifies a Payment."""


class RefundId(_BaseId):
    """Identifies a Refund."""


class EventId(_BaseId):
    """
    Identifies a domain event.

    Used as the idempotency key: if two events share the same EventId,
    the second is a duplicate and must not be processed again by the
    financial state engine.
    """


class DeviceId(_BaseId):
    """
    Identifies a device fingerprint.

    Used to build graph edges between customers sharing a device.
    Derived from hashed device attributes — never the raw fingerprint string.
    """


class AddressId(_BaseId):
    """
    Identifies a normalised shipping address.

    Derived from hashed address components. Two addresses are considered
    the same entity if their AddressId matches, regardless of formatting
    differences in the original string.
    """


class InstrumentId(_BaseId):
    """Identifies a payment instrument (card, UPI VPA, wallet, etc.)."""


class SessionId(_BaseId):
    """
    Identifies a browser or app session at payment time.

    Optional. Present only when the payment gateway supplies a session token.
    Used as a supplementary graph edge signal.
    """