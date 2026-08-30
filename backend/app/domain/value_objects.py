"""
Value objects for Refund Sentinel.

Value objects are immutable, equality-by-value types with no identity of their
own. They encapsulate domain rules about what constitutes a valid value.

Key types
---------
Money         — integer paise + explicit currency. No floats. Ever.
UTCDateTime   — timezone-aware datetime enforcing UTC.
UntrustedText — customer-supplied free text, explicitly marked as untrusted.
IpIdentifier  — network-level identifier for graph edges, validated via the
                standard-library ipaddress module.
"""

from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel, field_validator

from backend.app.domain.enums import Currency


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


class Money(BaseModel):
    """
    Represents a monetary amount in the smallest currency unit (paise for INR).

    Rules
    -----
    - amount_paise is a non-negative integer. Zero is permitted (e.g. fully
      waived items). Negative values are always rejected.
    - currency is always explicit. No implicit INR assumption.
    - Two Money values are equal only if both amount AND currency match.
    - Arithmetic is only permitted between same-currency amounts.
    - No floating-point representation at any point. The validator rejects
      floats explicitly; relying on Pydantic's int coercion (which would
      silently truncate 1.9 to 1) is not acceptable for financial amounts.

    Internal storage: amount_paise (int).
    Display conversion to rupees is the responsibility of the presentation
    layer, never the domain layer.
    """

    model_config = {"frozen": True}

    amount_paise: int
    currency: Currency

    @field_validator("amount_paise", mode="before")
    @classmethod
    def amount_must_be_non_negative_integer(cls, v: object) -> int:
        # Explicitly reject floats before Pydantic's coercion can silently truncate.
        if isinstance(v, float):
            raise TypeError(
                "amount_paise must be an integer, not a float. "
                "Do not use floats for monetary amounts. "
                f"Got: {v!r}"
            )
        if not isinstance(v, int):
            raise TypeError(
                f"amount_paise must be an integer, got {type(v).__name__!r}."
            )
        if v < 0:
            raise ValueError(
                f"amount_paise must be non-negative, got {v}. "
                "Negative monetary amounts are not permitted in the domain model."
            )
        return v

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(amount_paise=self.amount_paise + other.amount_paise, currency=self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        result = self.amount_paise - other.amount_paise
        if result < 0:
            raise ValueError(
                f"Subtraction would produce a negative monetary amount: "
                f"{self.amount_paise} - {other.amount_paise} = {result}. "
                "Domain invariant violated."
            )
        return Money(amount_paise=result, currency=self.currency)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __le__(self, other: "Money") -> bool:
        self._assert_same_currency(other)
        return self.amount_paise <= other.amount_paise

    def __lt__(self, other: "Money") -> bool:
        self._assert_same_currency(other)
        return self.amount_paise < other.amount_paise

    def __ge__(self, other: "Money") -> bool:
        self._assert_same_currency(other)
        return self.amount_paise >= other.amount_paise

    def __gt__(self, other: "Money") -> bool:
        self._assert_same_currency(other)
        return self.amount_paise > other.amount_paise

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_zero(self) -> bool:
        return self.amount_paise == 0

    def _assert_same_currency(self, other: "Money") -> None:
        if not isinstance(other, Money):
            raise TypeError(
                f"Cannot compare Money with {type(other).__name__!r}."
            )
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot operate on Money values with different currencies: "
                f"{self.currency.value} vs {other.currency.value}."
            )

    @classmethod
    def zero(cls, currency: Currency) -> "Money":
        return cls(amount_paise=0, currency=currency)

    @classmethod
    def of_paise(cls, amount: int, currency: Currency = Currency.INR) -> "Money":
        """Convenience constructor. amount must be a non-negative integer."""
        return cls(amount_paise=amount, currency=currency)

    def __repr__(self) -> str:
        return f"Money(amount_paise={self.amount_paise}, currency={self.currency.value!r})"


# ---------------------------------------------------------------------------
# UTCDateTime
# ---------------------------------------------------------------------------


class UTCDateTime(BaseModel):
    """
    A timezone-aware datetime that is always stored in UTC.

    Naive datetimes are rejected. Datetimes in non-UTC timezones are
    converted to UTC at construction.

    Canonical ordering key
    ----------------------
    occurred_at on a domain event carries the business occurrence time as
    reported by the originating system. The financial state engine (Phase 2)
    uses occurred_at — not received_at — to reconstruct the canonical order
    of events for a given payment. This distinction is critical: a batch of
    simulator events generated and ingested at the same wall-clock time may
    have occurred_at values spread across hours or days. Sorting by
    received_at would produce an incorrect order.

    received_at is ingestion provenance. It records when our system saw
    the event. It is used for audit and anomaly detection, not for
    business ordering.
    """

    model_config = {"frozen": True}

    value: datetime

    @field_validator("value", mode="before")
    @classmethod
    def must_be_utc_aware(cls, v: object) -> datetime:
        if not isinstance(v, datetime):
            raise TypeError(f"Expected a datetime instance, got {type(v).__name__!r}.")
        if v.tzinfo is None:
            raise ValueError(
                "Naive datetime rejected. All timestamps in Refund Sentinel must be "
                "timezone-aware UTC datetimes. "
                "Use datetime.now(timezone.utc) or attach tzinfo=timezone.utc."
            )
        # Normalise any timezone to UTC.
        return v.astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __le__(self, other: "UTCDateTime") -> bool:
        return self.value <= other.value

    def __lt__(self, other: "UTCDateTime") -> bool:
        return self.value < other.value

    def __ge__(self, other: "UTCDateTime") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "UTCDateTime") -> bool:
        return self.value > other.value

    # ------------------------------------------------------------------
    # Interval computation
    # ------------------------------------------------------------------

    def seconds_since(self, earlier: "UTCDateTime") -> float:
        """
        Returns the number of elapsed seconds between an earlier timestamp
        and this one.

        Raises ValueError if 'earlier' is actually after self, because the
        method name implies a specific temporal direction.
        """
        delta = self.value - earlier.value
        if delta.total_seconds() < 0:
            raise ValueError(
                f"seconds_since() called with a later timestamp as 'earlier': "
                f"self={self.value.isoformat()}, earlier={earlier.value.isoformat()}. "
                "Swap the arguments or use the absolute difference directly."
            )
        return delta.total_seconds()

    def hours_since(self, earlier: "UTCDateTime") -> float:
        """Returns elapsed hours. See seconds_since() for ordering rules."""
        return self.seconds_since(earlier) / 3600.0

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def now(cls) -> "UTCDateTime":
        return cls(value=datetime.now(timezone.utc))

    @classmethod
    def from_isoformat(cls, s: str) -> "UTCDateTime":
        """
        Parse an ISO 8601 string. The string must include a timezone offset.
        Naive ISO strings (no offset) are rejected.
        """
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse UTCDateTime from {s!r}: {exc}"
            ) from exc
        return cls(value=dt)

    def __repr__(self) -> str:
        return f"UTCDateTime({self.value.isoformat()})"


# ---------------------------------------------------------------------------
# UntrustedText
# ---------------------------------------------------------------------------

# Maximum length for any customer-supplied free-text field.
_MAX_UNTRUSTED_TEXT_LENGTH: Final[int] = 500


class UntrustedText(BaseModel):
    """
    Customer-supplied free text that must never be treated as a trusted signal.

    This type makes the untrusted nature explicit in the type system rather
    than relying on documentation or convention.

    It is used for fields such as refund reason text, which a customer can
    set to any content — including text designed to manipulate downstream
    ML feature pipelines or LLM prompt construction.

    Rules
    -----
    - Maximum 500 characters (see _MAX_UNTRUSTED_TEXT_LENGTH).
    - Stored as-is. No sanitisation at the domain layer. Sanitisation is the
      responsibility of the layer that consumes the text (e.g. the LLM
      prompt builder in Phase 7).
    - Must not be used as a raw feature input to the ML model.
    - Must be passed to the LLM only as a clearly-labelled data field,
      never interpolated into system-prompt instructions.

    The name UntrustedText (not UntrustableText) reflects the actual state
    of the data: it IS untrusted, not merely capable of being untrusted.
    """

    model_config = {"frozen": True}

    raw: str

    @field_validator("raw")
    @classmethod
    def enforce_length_limit(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(
                f"UntrustedText requires a string, got {type(v).__name__!r}."
            )
        if len(v) > _MAX_UNTRUSTED_TEXT_LENGTH:
            raise ValueError(
                f"Customer-supplied text exceeds the maximum permitted length of "
                f"{_MAX_UNTRUSTED_TEXT_LENGTH} characters (received {len(v)} characters)."
            )
        return v

    def __repr__(self) -> str:
        preview = self.raw[:40] + "..." if len(self.raw) > 40 else self.raw
        return f"UntrustedText({preview!r})"


# ---------------------------------------------------------------------------
# IpIdentifier
# ---------------------------------------------------------------------------


class IpIdentifier(BaseModel):
    """
    A network-level IP address identifier used for graph edge construction.

    Validation uses Python's standard-library ipaddress module, which
    correctly handles all valid IPv4 and IPv6 formats and rejects malformed
    inputs such as 999.999.999.999 or partial addresses.

    Important
    ---------
    This identifier is NOT a high-confidence fraud signal on its own.
    NAT, shared office networks, and mobile carrier networks routinely
    cause many legitimate customers to share an IP address. It is one
    weak structural signal among several, and its contribution to risk
    scoring is gated by the behavioral confirmation mechanism (Phase 4).

    Stored as a normalised string produced by str(ipaddress.ip_address(v)),
    which gives consistent representation (e.g. IPv6 is lower-cased and
    compressed).
    """

    model_config = {"frozen": True}

    value: str

    @field_validator("value", mode="before")
    @classmethod
    def must_be_valid_ip_address(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(
                f"IpIdentifier requires a string, got {type(v).__name__!r}."
            )
        stripped = v.strip()
        if not stripped:
            raise ValueError("IpIdentifier cannot be an empty string.")
        try:
            parsed = ipaddress.ip_address(stripped)
        except ValueError as exc:
            raise ValueError(
                f"IpIdentifier {v!r} is not a valid IPv4 or IPv6 address: {exc}"
            ) from exc
        # Return the canonical normalised form.
        return str(parsed)

    def __repr__(self) -> str:
        return f"IpIdentifier({self.value!r})"