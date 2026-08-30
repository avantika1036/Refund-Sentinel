"""
Tests for backend.app.domain.value_objects

Proves
------
Money:
  - Rejects floats explicitly (does not silently truncate).
  - Rejects negative amounts.
  - Accepts zero.
  - Addition and subtraction are correct and produce new instances.
  - Subtraction rejects results that would go negative.
  - _assert_same_currency does not raise when currencies match.
  - _assert_same_currency raises TypeError when passed a non-Money object.
  - _assert_same_currency currency-mismatch branch: see note below.
  - Comparison operators work correctly for same-currency values.
  - Frozen after construction.

Note on currency-mismatch branch coverage
------------------------------------------
Money._assert_same_currency checks isinstance(other, Money) first. A real
Money instance always has currency=Currency.INR because Currency currently
defines only one member. Therefore the branch:

    if self.currency != other.currency: raise ValueError(...)

is structurally unreachable in tests without either adding a second Currency
member or bypassing Pydantic's frozen model to corrupt an existing Money
instance — both of which are explicitly disallowed by the project contract.

The test suite covers:
  (a) the positive case — same currency, no error;
  (b) the TypeError guard — non-Money argument rejected before the currency
      comparison is reached.

If a second currency is added in a future phase, a currency-mismatch test
must be added at that point.

UTCDateTime:
  - Rejects naive datetimes.
  - Normalises non-UTC timezones to UTC.
  - from_isoformat() accepts timezone-aware ISO strings.
  - from_isoformat() rejects naive ISO strings.
  - Ordering operators work correctly.
  - seconds_since() computes correctly and rejects incorrect argument order.
  - hours_since() computes correctly.
  - Frozen after construction.

UntrustedText:
  - Accepts short text and empty string.
  - Accepts text at exactly the length limit.
  - Rejects text exceeding the length limit.
  - Rejects non-string input.
  - Frozen after construction.
  - repr() truncates long text; does not truncate short text.

IpIdentifier:
  - Accepts valid IPv4 and IPv6 addresses.
  - Rejects malformed IPv4 (999.999.999.999).
  - Rejects partial IPv4.
  - Rejects hostnames, empty strings, and whitespace-only strings.
  - Normalises IPv6 to compressed form consistently.
  - Frozen after construction.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.domain.enums import Currency
from backend.app.domain.value_objects import (
    IpIdentifier,
    Money,
    UTCDateTime,
    UntrustedText,
)


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


class TestMoney:
    def test_valid_construction(self):
        m = Money(amount_paise=100, currency=Currency.INR)
        assert m.amount_paise == 100
        assert m.currency == Currency.INR

    def test_zero_is_valid(self):
        m = Money.zero(Currency.INR)
        assert m.amount_paise == 0
        assert m.is_zero()

    def test_rejects_negative_amount(self):
        with pytest.raises(ValueError, match="non-negative"):
            Money(amount_paise=-1, currency=Currency.INR)

    def test_rejects_float_explicitly(self):
        """
        Floats must be rejected with TypeError, not silently truncated.
        1.9 must not become 1 paise.
        """
        with pytest.raises(TypeError, match="float"):
            Money(amount_paise=1.9, currency=Currency.INR)

    def test_rejects_float_zero(self):
        """0.0 is a float and must be rejected even though its numeric value is zero."""
        with pytest.raises(TypeError, match="float"):
            Money(amount_paise=0.0, currency=Currency.INR)

    def test_rejects_string_amount(self):
        with pytest.raises((TypeError, Exception)):
            Money(amount_paise="100", currency=Currency.INR)

    def test_of_paise_convenience_constructor(self):
        m = Money.of_paise(5000)
        assert m.amount_paise == 5000
        assert m.currency == Currency.INR

    def test_addition_correct(self):
        a = Money.of_paise(100)
        b = Money.of_paise(200)
        assert (a + b).amount_paise == 300

    def test_addition_produces_new_instance(self):
        """Addition must not mutate either operand."""
        a = Money.of_paise(100)
        b = Money.of_paise(200)
        result = a + b
        assert result is not a
        assert result is not b
        assert a.amount_paise == 100
        assert b.amount_paise == 200

    def test_subtraction_correct(self):
        a = Money.of_paise(500)
        b = Money.of_paise(200)
        assert (a - b).amount_paise == 300

    def test_subtraction_to_zero(self):
        a = Money.of_paise(100)
        b = Money.of_paise(100)
        result = a - b
        assert result.amount_paise == 0
        assert result.is_zero()

    def test_subtraction_rejects_negative_result(self):
        a = Money.of_paise(100)
        b = Money.of_paise(200)
        with pytest.raises(ValueError, match="negative"):
            _ = a - b

    def test_currency_guard_does_not_raise_for_same_currency(self):
        """
        _assert_same_currency must not raise when both operands have the
        same currency. This is the normal production path.
        """
        a = Money.of_paise(100, Currency.INR)
        b = Money.of_paise(200, Currency.INR)
        a._assert_same_currency(b)  # Must not raise.

    def test_currency_guard_raises_type_error_for_non_money_argument(self):
        """
        _assert_same_currency raises TypeError when passed an argument that
        is not a Money instance. This guards against programming errors where
        the wrong type is passed to an arithmetic or comparison method.

        Note: this tests the isinstance guard, not the currency-mismatch branch.
        The currency-mismatch branch (self.currency != other.currency) is
        structurally unreachable while Currency has exactly one member (INR)
        and Money's Pydantic validator enforces that all Money instances carry
        a Currency enum value. See module docstring for full explanation.
        """
        a = Money.of_paise(100)
        with pytest.raises(TypeError):
            a._assert_same_currency(100)  # type: ignore[arg-type]

    def test_currency_guard_raises_type_error_for_plain_string_argument(self):
        """_assert_same_currency raises TypeError when passed a plain string."""
        a = Money.of_paise(100)
        with pytest.raises(TypeError):
            a._assert_same_currency("100 INR")  # type: ignore[arg-type]

    def test_comparison_less_than(self):
        assert Money.of_paise(100) < Money.of_paise(200)

    def test_comparison_greater_than(self):
        assert Money.of_paise(200) > Money.of_paise(100)

    def test_comparison_less_than_or_equal_equal_values(self):
        assert Money.of_paise(100) <= Money.of_paise(100)

    def test_comparison_less_than_or_equal_lesser_value(self):
        assert Money.of_paise(99) <= Money.of_paise(100)

    def test_comparison_greater_than_or_equal_equal_values(self):
        assert Money.of_paise(100) >= Money.of_paise(100)

    def test_comparison_greater_than_or_equal_greater_value(self):
        assert Money.of_paise(101) >= Money.of_paise(100)

    def test_frozen(self):
        m = Money.of_paise(100)
        with pytest.raises(Exception):
            m.amount_paise = 999

    def test_repr_contains_amount_and_currency(self):
        m = Money.of_paise(100)
        r = repr(m)
        assert "100" in r
        assert "INR" in r


# ---------------------------------------------------------------------------
# UTCDateTime
# ---------------------------------------------------------------------------


class TestUTCDateTime:
    def test_valid_utc_datetime(self):
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        utc = UTCDateTime(value=dt)
        assert utc.value.tzinfo == timezone.utc

    def test_rejects_naive_datetime(self):
        naive = datetime(2024, 6, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="Naive datetime"):
            UTCDateTime(value=naive)

    def test_normalises_non_utc_to_utc(self):
        """A timezone-aware datetime in IST (+05:30) is converted to UTC."""
        ist = timezone(timedelta(hours=5, minutes=30))
        dt_ist = datetime(2024, 6, 1, 17, 30, 0, tzinfo=ist)
        utc = UTCDateTime(value=dt_ist)
        # 17:30 IST = 12:00 UTC
        assert utc.value.hour == 12
        assert utc.value.minute == 0
        assert utc.value.tzinfo == timezone.utc

    def test_from_isoformat_valid_with_offset(self):
        utc = UTCDateTime.from_isoformat("2024-06-01T12:00:00+00:00")
        assert utc.value.year == 2024
        assert utc.value.tzinfo is not None

    def test_from_isoformat_rejects_naive_iso_string(self):
        """An ISO 8601 string without a timezone offset is a naive datetime."""
        with pytest.raises(ValueError):
            UTCDateTime.from_isoformat("2024-06-01T12:00:00")

    def test_ordering_less_than(self):
        earlier = UTCDateTime.from_isoformat("2024-06-01T10:00:00+00:00")
        later = UTCDateTime.from_isoformat("2024-06-01T12:00:00+00:00")
        assert earlier < later

    def test_ordering_greater_than(self):
        earlier = UTCDateTime.from_isoformat("2024-06-01T10:00:00+00:00")
        later = UTCDateTime.from_isoformat("2024-06-01T12:00:00+00:00")
        assert later > earlier

    def test_seconds_since_correct(self):
        earlier = UTCDateTime.from_isoformat("2024-06-01T10:00:00+00:00")
        later = UTCDateTime.from_isoformat("2024-06-01T10:01:00+00:00")
        assert later.seconds_since(earlier) == 60.0

    def test_seconds_since_rejects_reversed_arguments(self):
        """seconds_since raises ValueError when 'earlier' is actually after self."""
        earlier = UTCDateTime.from_isoformat("2024-06-01T10:00:00+00:00")
        later = UTCDateTime.from_isoformat("2024-06-01T10:01:00+00:00")
        with pytest.raises(ValueError):
            earlier.seconds_since(later)

    def test_hours_since_correct(self):
        earlier = UTCDateTime.from_isoformat("2024-06-01T10:00:00+00:00")
        later = UTCDateTime.from_isoformat("2024-06-01T12:00:00+00:00")
        assert later.hours_since(earlier) == 2.0

    def test_frozen(self):
        utc = UTCDateTime.now()
        with pytest.raises(Exception):
            utc.value = datetime.now(timezone.utc)

    def test_now_produces_utc_aware_datetime(self):
        utc = UTCDateTime.now()
        assert utc.value.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# UntrustedText
# ---------------------------------------------------------------------------


class TestUntrustedText:
    def test_accepts_valid_short_text(self):
        t = UntrustedText(raw="Item arrived damaged.")
        assert t.raw == "Item arrived damaged."

    def test_accepts_empty_string(self):
        """Empty string is permitted — customer may omit the reason description."""
        t = UntrustedText(raw="")
        assert t.raw == ""

    def test_accepts_text_at_exact_limit(self):
        t = UntrustedText(raw="x" * 500)
        assert len(t.raw) == 500

    def test_rejects_text_exceeding_limit(self):
        with pytest.raises(ValueError, match="maximum"):
            UntrustedText(raw="x" * 501)

    def test_rejects_non_string(self):
        with pytest.raises((TypeError, Exception)):
            UntrustedText(raw=42)  # type: ignore[arg-type]

    def test_frozen(self):
        t = UntrustedText(raw="hello")
        with pytest.raises(Exception):
            t.raw = "changed"

    def test_repr_truncates_long_text(self):
        t = UntrustedText(raw="a" * 100)
        r = repr(t)
        assert "..." in r

    def test_repr_does_not_truncate_short_text(self):
        t = UntrustedText(raw="short")
        r = repr(t)
        assert "..." not in r


# ---------------------------------------------------------------------------
# IpIdentifier
# ---------------------------------------------------------------------------


class TestIpIdentifier:
    def test_accepts_valid_ipv4(self):
        ip = IpIdentifier(value="192.168.1.1")
        assert ip.value == "192.168.1.1"

    def test_accepts_valid_ipv4_with_surrounding_whitespace(self):
        """Leading and trailing whitespace is stripped before validation."""
        ip = IpIdentifier(value="  10.0.0.1  ")
        assert ip.value == "10.0.0.1"

    def test_accepts_loopback_ipv4(self):
        ip = IpIdentifier(value="127.0.0.1")
        assert ip.value == "127.0.0.1"

    def test_accepts_valid_ipv6_compressed(self):
        ip = IpIdentifier(value="2001:db8::1")
        assert "2001" in ip.value

    def test_accepts_loopback_ipv6(self):
        ip = IpIdentifier(value="::1")
        assert ip.value == "::1"

    def test_rejects_malformed_ipv4_octets_out_of_range(self):
        """999.999.999.999 must be rejected — octets exceed 255."""
        with pytest.raises(ValueError):
            IpIdentifier(value="999.999.999.999")

    def test_rejects_partial_ipv4(self):
        """192.168.1 is incomplete and must be rejected."""
        with pytest.raises(ValueError):
            IpIdentifier(value="192.168.1")

    def test_rejects_hostname(self):
        with pytest.raises(ValueError):
            IpIdentifier(value="example.com")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            IpIdentifier(value="")

    def test_rejects_whitespace_only_string(self):
        with pytest.raises(ValueError, match="empty"):
            IpIdentifier(value="   ")

    def test_rejects_non_string_input(self):
        with pytest.raises((TypeError, Exception)):
            IpIdentifier(value=192168)  # type: ignore[arg-type]

    def test_frozen(self):
        ip = IpIdentifier(value="10.0.0.1")
        with pytest.raises(Exception):
            ip.value = "10.0.0.2"

    def test_ipv6_normalisation_is_consistent(self):
        """
        Full and compressed representations of the same IPv6 address must
        produce the same normalised value. The ipaddress module handles this.
        """
        full = IpIdentifier(value="2001:0db8:0000:0000:0000:0000:0000:0001")
        compressed = IpIdentifier(value="2001:db8::1")
        assert full.value == compressed.value