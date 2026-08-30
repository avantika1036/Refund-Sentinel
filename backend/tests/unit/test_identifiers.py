"""
Tests for backend.app.domain.identifiers

Proves
------
- Each ID type constructs correctly from a uuid.UUID.
- from_str() parses valid UUID strings.
- from_str() raises ValueError on invalid strings and empty strings.
- Passing a non-UUID to the constructor raises TypeError.
- IDs are immutable after construction.
- Two IDs of the same type with the same value are equal.
- Two IDs of different types with the same value are NOT equal.
- IDs are hashable and usable in sets and as dict keys.
- Different ID types with the same UUID hash to different values.
- generate() produces a valid, unique ID.
- str() returns the lowercase hyphenated UUID string.
- repr() contains the class name.
- Pydantic rejects a sibling ID type at model construction time.
"""

import uuid

import pytest
from pydantic import BaseModel, ValidationError

from backend.app.domain.identifiers import (
    AddressId,
    CustomerId,
    DeviceId,
    EventId,
    MerchantId,
    OrderId,
    PaymentId,
    RefundId,
    SessionId,
)

_ALL_ID_CLASSES = [
    MerchantId,
    CustomerId,
    OrderId,
    PaymentId,
    RefundId,
    EventId,
    DeviceId,
    AddressId,
    SessionId,
]

_SAMPLE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_construct_from_uuid(cls):
    """Each ID type accepts a uuid.UUID instance."""
    uid = uuid.uuid4()
    instance = cls(uid)
    assert instance.value == uid


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_construct_rejects_plain_string(cls):
    """Passing a plain string (not a uuid.UUID) to the constructor raises TypeError."""
    with pytest.raises(TypeError, match="uuid.UUID"):
        cls("12345678-1234-5678-1234-567812345678")


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_construct_rejects_integer(cls):
    """Passing an integer raises TypeError."""
    with pytest.raises(TypeError):
        cls(42)


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_from_str_valid_uuid(cls):
    """from_str() parses a valid UUID string and produces the correct value."""
    raw = "12345678-1234-5678-1234-567812345678"
    instance = cls.from_str(raw)
    assert str(instance) == raw


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_from_str_invalid_string(cls):
    """from_str() raises ValueError for a non-UUID string."""
    with pytest.raises(ValueError):
        cls.from_str("not-a-uuid")


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_from_str_empty_string(cls):
    """from_str() raises ValueError for an empty string."""
    with pytest.raises(ValueError):
        cls.from_str("")


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_immutable_after_construction(cls):
    """Attempting to set any attribute after construction raises AttributeError."""
    instance = cls.generate()
    with pytest.raises(AttributeError):
        instance._value = uuid.uuid4()


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_equal_same_type_same_value(cls):
    """Two IDs of the same type with the same UUID value are equal."""
    uid = uuid.uuid4()
    assert cls(uid) == cls(uid)


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_not_equal_same_type_different_value(cls):
    """Two IDs of the same type with different UUIDs are not equal."""
    assert cls(uuid.uuid4()) != cls(uuid.uuid4())


def test_not_equal_different_types_same_uuid():
    """PaymentId and RefundId with the same UUID are not equal."""
    uid = _SAMPLE_UUID
    assert PaymentId(uid) != RefundId(uid)


def test_cross_type_equality_returns_not_implemented():
    """
    __eq__ between different ID types returns NotImplemented,
    allowing Python to fall back to identity comparison.
    """
    uid = _SAMPLE_UUID
    result = PaymentId(uid).__eq__(RefundId(uid))
    assert result is NotImplemented


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_hashable_can_be_placed_in_set(cls):
    """Two identical IDs occupy one slot in a set."""
    uid = uuid.uuid4()
    s = {cls(uid), cls(uid)}
    assert len(s) == 1


def test_different_types_produce_different_hashes():
    """PaymentId and RefundId with the same UUID hash to different values."""
    uid = _SAMPLE_UUID
    assert hash(PaymentId(uid)) != hash(RefundId(uid))


def test_usable_as_dict_key():
    """An ID can be used as a dictionary key and retrieved correctly."""
    pid = PaymentId.generate()
    d = {pid: "payment-data"}
    assert d[pid] == "payment-data"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_generate_returns_valid_id(cls):
    """generate() returns an instance of the correct type with a UUID value."""
    instance = cls.generate()
    assert isinstance(instance.value, uuid.UUID)
    assert isinstance(instance, cls)


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_generate_returns_unique_ids(cls):
    """Two successive calls to generate() return different IDs."""
    assert cls.generate() != cls.generate()


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_str_produces_uuid_string(cls):
    """str(id) produces the lowercase hyphenated UUID string."""
    uid = _SAMPLE_UUID
    assert str(cls(uid)) == str(uid)


@pytest.mark.parametrize("cls", _ALL_ID_CLASSES)
def test_repr_contains_class_name(cls):
    """repr(id) includes the concrete class name."""
    instance = cls.generate()
    assert cls.__name__ in repr(instance)


# ---------------------------------------------------------------------------
# Pydantic cross-type rejection
# ---------------------------------------------------------------------------


class _PaymentModel(BaseModel):
    """Minimal Pydantic model used to test cross-type ID rejection."""
    payment_id: PaymentId
    refund_id: RefundId


def test_pydantic_rejects_refund_id_where_payment_id_expected():
    """
    Pydantic must raise ValidationError when a RefundId is supplied
    for a field typed PaymentId.

    This is the core guarantee of the typed ID system: cross-type
    assignment is caught at validation time, not silently accepted.
    """
    with pytest.raises(ValidationError):
        _PaymentModel(
            payment_id=RefundId.generate(),  # wrong type
            refund_id=RefundId.generate(),
        )


def test_pydantic_rejects_payment_id_where_refund_id_expected():
    """
    Pydantic must raise ValidationError when a PaymentId is supplied
    for a field typed RefundId.
    """
    with pytest.raises(ValidationError):
        _PaymentModel(
            payment_id=PaymentId.generate(),
            refund_id=PaymentId.generate(),  # wrong type
        )


def test_pydantic_accepts_correct_types():
    """Pydantic accepts the correct ID types without error."""
    model = _PaymentModel(
        payment_id=PaymentId.generate(),
        refund_id=RefundId.generate(),
    )
    assert isinstance(model.payment_id, PaymentId)
    assert isinstance(model.refund_id, RefundId)


def test_pydantic_accepts_uuid_string_for_id_field():
    """
    Pydantic accepts a raw UUID string for an ID field and constructs
    the correct typed ID from it.
    """
    raw = str(uuid.uuid4())
    model = _PaymentModel(
        payment_id=raw,
        refund_id=str(uuid.uuid4()),
    )
    assert isinstance(model.payment_id, PaymentId)


def test_pydantic_accepts_uuid_object_for_id_field():
    """
    Pydantic accepts a uuid.UUID object for an ID field and wraps it
    in the correct typed ID class.
    """
    uid = uuid.uuid4()
    model = _PaymentModel(
        payment_id=uid,
        refund_id=uuid.uuid4(),
    )
    assert isinstance(model.payment_id, PaymentId)
    assert model.payment_id.value == uid