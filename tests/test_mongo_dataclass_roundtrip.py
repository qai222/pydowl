from __future__ import annotations

from typing import Optional


from pydowl import PydOwlClass, PydOwlDataClass, PydOwlRegistry


def reset_registry() -> None:
    PydOwlRegistry.clear()


def test_pydowl_dataclass_roundtrip_to_from_mongo_docs() -> None:
    """
    PydOwlDataClass value object should round-trip cleanly via
    to_mongo_docs / from_mongo_docs, with identity fields populated
    at construction time (not mutated after).
    """

    reset_registry()

    class Amount(PydOwlDataClass):
        # Explicit identity fields to make sure our from_mongo_docs
        # path sets them once and never mutates them via .update().
        __id_fields__ = ("value", "unit")

        value: Optional[float] = None
        unit: Optional[str] = None
        label: Optional[str] = None

    # No forward refs, so no model_rebuild() needed.
    a1 = Amount(value=1.5, unit="g", label="mass")

    docs = a1.to_mongo_docs()
    assert len(docs) == 1
    root_id = a1.identifier

    reset_registry()
    a2 = Amount.from_mongo_docs(root_id, docs)

    # Basic equality
    assert isinstance(a2, Amount)
    assert a2.identifier == a1.identifier
    assert a2.value == a1.value
    assert a2.unit == a1.unit
    assert a2.label == a1.label

    # Ensure we did *not* mutate identity fields via .update()
    # (if we had, this would have raised TypeError inside from_mongo_docs).
    assert a2.__class__ is Amount


def test_pydowl_class_roundtrip_to_from_mongo_docs_simple() -> None:
    """
    Simple PydOwlClass graph (no dataclasses) should round-trip via
    to_mongo_docs / from_mongo_docs and preserve cycles / sharing.
    """

    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    alice = Person(identifier="alice", name="Alice")
    bob = Person(identifier="bob", name="Bob")
    alice.spouse = bob
    bob.spouse = alice

    docs = alice.to_mongo_docs()
    # Two nodes: alice + bob
    assert {d["_id"] for d in docs} == {"alice", "bob"}

    reset_registry()
    alice2 = Person.from_mongo_docs("alice", docs)

    assert isinstance(alice2, Person)
    assert alice2.name == "Alice"
    assert isinstance(alice2.spouse, Person)
    assert alice2.spouse.name == "Bob"
    # Cycle preserved
    assert alice2.spouse.spouse is alice2


def test_nested_pydowl_class_with_dataclass_field_roundtrip() -> None:
    """
    Nested case: a PydOwlClass (Order) containing a PydOwlDataClass (Price)
    as a field should round-trip via to_mongo_docs / from_mongo_docs.

    This exercises the branch in from_mongo_docs that builds dataclasses
    in one shot while still using the placeholder+update path for the
    owning PydOwlClass, and ensures the graph is constructed correctly.
    """

    reset_registry()

    class Price(PydOwlDataClass):
        __id_fields__ = ("amount", "currency")

        amount: Optional[float] = None
        currency: Optional[str] = None

    class Order(PydOwlClass):
        code: Optional[str] = None
        price: Optional[Price] = None

    # No forward refs; model_rebuild() not strictly required here.
    p1 = Price(amount=9.99, currency="USD")
    o1 = Order(identifier="order-1", code="O-001", price=p1)

    docs = o1.to_mongo_docs()
    ids = {d["_id"] for d in docs}
    assert "order-1" in ids
    assert p1.identifier in ids

    reset_registry()
    o2 = Order.from_mongo_docs("order-1", docs)

    assert isinstance(o2, Order)
    assert o2.identifier == "order-1"
    assert o2.code == "O-001"
    assert isinstance(o2.price, Price)
    assert o2.price.amount == 9.99
    assert o2.price.currency == "USD"

    # Dataclass instance is a fresh object but preserves its identifier
    assert o2.price is not p1
    assert o2.price.identifier == p1.identifier
