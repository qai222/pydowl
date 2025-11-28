from __future__ import annotations

import json
import string
from typing import Optional, List

import owlready2
import pytest

from pydowl import (
    PydOwlClass,
    PydOwlDataClass,
    PydOwlRegistry,
)
from pydowl.base_class import KgPullError


def reset_registry() -> None:
    PydOwlRegistry.clear()


# ──────────────────────────────────────────────────────────────────────
# F1 – class → collection mapping
# ──────────────────────────────────────────────────────────────────────


def test_mongo_collection_name_default_and_override():
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None

    class WeirdThing(PydOwlClass):
        __mongo_collection__ = "weirdos"
        label: Optional[str] = None

    assert Person.mongo_collection_name() == "persons"
    assert WeirdThing.mongo_collection_name() == "weirdos"


# ──────────────────────────────────────────────────────────────────────
# F2/F3 – to_mongo_docs and from_mongo_docs (tree + cycle)
# ──────────────────────────────────────────────────────────────────────


def test_to_mongo_docs_and_from_mongo_docs_roundtrip_tree():
    """
    Simple tree-shaped graph should round-trip via to_mongo_docs /
    from_mongo_docs and preserve identifiers and relationships.
    """
    reset_registry()

    class Address(PydOwlClass):
        city: Optional[str] = None

    class Person(PydOwlClass):
        name: Optional[str] = None
        address: Optional[Address] = None
        friends: List["Person"] = []

    Person.model_rebuild()

    addr = Address(identifier="addr1", city="London")
    bob = Person(identifier="bob", name="Bob", address=None, friends=[])
    alice = Person(identifier="alice", name="Alice", address=addr, friends=[bob])

    docs = alice.to_mongo_docs()

    ids = sorted(d["_id"] for d in docs)
    assert ids == ["addr1", "alice", "bob"]

    alice_doc = next(d for d in docs if d["_id"] == "alice")
    bob_doc = next(d for d in docs if d["_id"] == "bob")
    addr_doc = next(d for d in docs if d["_id"] == "addr1")

    assert alice_doc["name"] == "Alice"
    assert alice_doc["address"] == "addr1"
    assert alice_doc["friends"] == ["bob"]

    assert bob_doc["name"] == "Bob"
    assert "address" not in bob_doc or bob_doc["address"] in (None, "")

    assert addr_doc["city"] == "London"

    reset_registry()
    alice2 = Person.from_mongo_docs("alice", docs)

    assert isinstance(alice2, Person)
    assert alice2.identifier == "alice"
    assert alice2.name == "Alice"
    assert isinstance(alice2.address, Address)
    assert alice2.address.city == "London"
    assert len(alice2.friends) == 1
    friend = alice2.friends[0]
    assert isinstance(friend, Person)
    assert friend.identifier == "bob"
    assert friend.name == "Bob"


def test_from_mongo_docs_handles_cycle_and_registry_reuse():
    """
    A simple cycle (alice <-> bob as spouse) should be reconstructed
    without infinite recursion, and shared instances should be reused.
    """
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    alice = Person(identifier="alice", name="Alice", spouse=None)
    bob = Person(identifier="bob", name="Bob", spouse=None)
    alice.spouse = bob
    bob.spouse = alice

    docs = alice.to_mongo_docs()
    reset_registry()

    alice2 = Person.from_mongo_docs("alice", docs)

    assert isinstance(alice2, Person)
    assert alice2.identifier == "alice"
    assert isinstance(alice2.spouse, Person)
    bob2 = alice2.spouse
    assert bob2.identifier == "bob"
    assert bob2.spouse is alice2


def test_from_mongo_docs_missing_id_raises():
    reset_registry()

    class Foo(PydOwlClass):
        name: Optional[str] = None

    docs = [{"name": "no_id"}]

    with pytest.raises(KgPullError):
        Foo.from_mongo_docs("whatever", docs)


def test_from_mongo_docs_missing_reference_raises():
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    docs = [
        {
            "_id": "alice",
            "pydowl_type": f"{Person.__module__}:{Person.__qualname__}",
            "name": "Alice",
            "spouse": "missing",
        }
    ]

    with pytest.raises(KgPullError):
        Person.from_mongo_docs("alice", docs)


# ──────────────────────────────────────────────────────────────────────
# G1/G2 – tree-safe JSON serialisation
# ──────────────────────────────────────────────────────────────────────


def test_to_tree_dict_for_tree():
    reset_registry()

    class Address(PydOwlClass):
        city: Optional[str] = None

    class Person(PydOwlClass):
        name: Optional[str] = None
        address: Optional[Address] = None

    addr = Address(identifier="addr1", city="Paris")
    alice = Person(identifier="alice", name="Alice", address=addr)

    tree = alice.to_tree_dict()

    assert tree["identifier"] == "alice"
    assert "pydowl_type" in tree
    mod, qual = tree["pydowl_type"].split(":", 1)
    assert qual.endswith("Person")

    assert tree["name"] == "Alice"
    assert "address" in tree
    assert tree["address"]["identifier"] == "addr1"
    assert tree["address"]["city"] == "Paris"


def test_to_tree_dict_raises_on_cycle():
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    alice = Person(identifier="alice", name="Alice", spouse=None)
    bob = Person(identifier="bob", name="Bob", spouse=None)
    alice.spouse = bob
    bob.spouse = alice

    with pytest.raises(ValueError):
        _ = alice.to_tree_dict()


def test_dump_tree_json_contains_identifier():
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None

    alice = Person(identifier="alice", name="Alice")

    s = alice.dump_tree_json()
    data = json.loads(s)
    assert data["identifier"] == "alice"
    assert data["name"] == "Alice"


# ──────────────────────────────────────────────────────────────────────
# H1 – model_json_schema for recursive models
# ──────────────────────────────────────────────────────────────────────


def test_model_json_schema_for_recursive_model_has_defs():
    """
    H1: Pydantic should be able to generate a JSON Schema for a
    recursive model (e.g. Person.spouse: Optional[Person]).
    """
    reset_registry()

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()
    schema = Person.model_json_schema()

    assert "$defs" in schema
    assert "Person" in schema["$defs"]
    schema_str = json.dumps(schema)
    assert "#/$defs/Person" in schema_str


# ──────────────────────────────────────────────────────────────────────
# J – PydOwlDataClass behaviour (stable id, immutability, format)
# ──────────────────────────────────────────────────────────────────────


def test_dataclass_stable_identifier_same_values():
    reset_registry()

    class ContinuousQuantity(PydOwlDataClass):
        value: Optional[float] = None
        unit: Optional[str] = None

    q1 = ContinuousQuantity(value=1.0, unit="kg")
    q2 = ContinuousQuantity(value=1.0, unit="kg")

    assert q1.identifier == q2.identifier
    assert q1 is not q2  # different Python objects


def test_dataclass_stable_identifier_diff_values():
    reset_registry()

    class ContinuousQuantity(PydOwlDataClass):
        value: Optional[float] = None
        unit: Optional[str] = None

    q1 = ContinuousQuantity(value=1.0, unit="kg")
    q2 = ContinuousQuantity(value=2.0, unit="kg")
    q3 = ContinuousQuantity(value=1.0, unit="m")

    assert q1.identifier != q2.identifier
    assert q1.identifier != q3.identifier
    assert q2.identifier != q3.identifier


def test_dataclass_identifier_has_pydowl_type_prefix():
    reset_registry()

    class ContinuousQuantity(PydOwlDataClass):
        value: Optional[float] = None
        unit: Optional[str] = None

    q = ContinuousQuantity(value=1.0, unit="kg")
    prefix = f"{ContinuousQuantity.__module__}:{ContinuousQuantity.__qualname__}:"
    assert q.identifier.startswith(prefix)
    suffix = q.identifier[len(prefix) :]
    assert len(suffix) == 16
    assert all(c in string.hexdigits for c in suffix)


def test_dataclass_respects_explicit_identifier():
    reset_registry()

    class ContinuousQuantity(PydOwlDataClass):
        value: Optional[float] = None
        unit: Optional[str] = None

    q = ContinuousQuantity(identifier="explicit_id", value=1.0, unit="kg")
    assert q.identifier == "explicit_id"

    # Identity fields are still immutable
    with pytest.raises(TypeError):
        q.value = 2.0  # type: ignore[assignment]


def test_dataclass_id_fields_immutable_but_other_fields_mutable():
    reset_registry()

    class WeightedQuantity(PydOwlDataClass):
        __id_fields__ = ("value", "unit")
        value: Optional[float] = None
        unit: Optional[str] = None
        note: Optional[str] = None

    w = WeightedQuantity(value=1.0, unit="kg", note="original")

    with pytest.raises(TypeError):
        w.value = 2.0  # type: ignore[assignment]

    with pytest.raises(TypeError):
        w.unit = "g"  # type: ignore[assignment]

    w.note = "updated"
    assert w.note == "updated"


def test_dataclass_to_mongo_docs_and_push_owlready_deduplicates():
    """
    Two PydOwlDataClass instances with the same value/unit should
    produce the same identifier and map to a single OWL individual.
    """
    reset_registry()

    class ContinuousQuantity(PydOwlDataClass):
        value: Optional[float] = None
        unit: Optional[str] = None

    q1 = ContinuousQuantity(value=1.0, unit="kg")
    q2 = ContinuousQuantity(value=1.0, unit="kg")

    assert q1.identifier == q2.identifier

    class Measurement(PydOwlClass):
        quantity: Optional[ContinuousQuantity] = None

    m1 = Measurement(identifier="m1", quantity=q1)
    docs = m1.to_mongo_docs()
    ids = sorted(d["_id"] for d in docs)
    assert "m1" in ids
    assert q1.identifier in ids

    onto = owlready2.get_ontology("http://example.org/dataclass_test#")
    ind1 = q1.push_owlready(onto, dynamic_tbox=True)
    ind2 = q2.push_owlready(onto, dynamic_tbox=True)

    assert ind1 is ind2
    iri = f"{onto.base_iri}{q1.identifier}"
    assert onto.search_one(iri=iri) is ind1


# ──────────────────────────────────────────────────────────────────────
# Registry + register_graph semantics (from old test_registry.py)
# ──────────────────────────────────────────────────────────────────────


class Bar(PydOwlClass):
    bar_field: Optional[str] = None


class Foo(PydOwlClass):
    foo_field: Optional[int] = 0
    bar: Optional[Bar] = None
    bars: List[Bar] = []


def test_singleton_semantics(test_onto):
    # Create two instances with the same identifier
    a = Bar(identifier="bar", bar_field="bar_field")
    PydOwlRegistry.register(a)

    a_ind = a.push_owlready(test_onto, dynamic_tbox=True)
    a_reg_rec = Bar.pull_owlready(test_onto, a_ind)
    assert a_reg_rec is a, "Registry should return the same object instance"


def test_nested_register_graph(test_onto):
    a = Bar(identifier="bar", bar_field="bar_field")
    a1 = Bar(identifier="a1", bar_field="bar_field")
    b = Foo(identifier="x", bar=a, bars=[a1, a])

    PydOwlRegistry.register_graph(b)

    b_ind = b.push_owlready(test_onto, dynamic_tbox=True)
    b_reg_rec = Foo.pull_owlready(test_onto, b_ind)

    assert b_reg_rec is b
    assert a is b_reg_rec.bar
    assert a1 is b_reg_rec.bars[0]
