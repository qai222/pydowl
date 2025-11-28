from __future__ import annotations

from typing import Optional

import owlready2
import pytest

from pydowl import PydOwlClass
from pydowl.base_class import KgPushError, KgPullError


# ──────────────────────────────────────────────────────────────────────
# D1 – cycle handling in _push_owlready
# ──────────────────────────────────────────────────────────────────────


def test_push_owlready_handles_simple_cycle():
    """
    D1: _push_owlready should handle cycles in the Python object graph
    by reusing existing OWL individuals via its visited map.

    For functional object properties, Owlready2 exposes the value as a
    single object (not a list), so we assert direct identity.
    """

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    alice = Person(identifier="alice", name="Alice", spouse=None)
    bob = Person(identifier="bob", name="Bob", spouse=None)
    alice.spouse = bob
    bob.spouse = alice

    onto = owlready2.get_ontology("http://example.org/cycle_test#")

    alice_ind = alice.push_owlready(onto, dynamic_tbox=True)
    assert alice_ind is not None

    alice_iri = f"{onto.base_iri}{alice.identifier}"
    bob_iri = f"{onto.base_iri}{bob.identifier}"

    alice_ind2 = onto.search_one(iri=alice_iri)
    bob_ind2 = onto.search_one(iri=bob_iri)

    assert alice_ind2 is not None
    assert bob_ind2 is not None
    assert len(onto.search(iri=alice_iri)) == 1
    assert len(onto.search(iri=bob_iri)) == 1

    assert hasattr(alice_ind2, "has_spouse")
    assert hasattr(bob_ind2, "has_spouse")
    assert alice_ind2.has_spouse is bob_ind2
    assert bob_ind2.has_spouse is alice_ind2


# ──────────────────────────────────────────────────────────────────────
# D3 – non-functional data properties should raise KgPullError
# ──────────────────────────────────────────────────────────────────────


def test_pull_owlready_errors_on_non_functional_data_property():
    """
    D3: If a data property is not a FunctionalProperty, pull_owlready
    should raise a KgPullError rather than silently picking a value.
    """

    class ExampleModel(PydOwlClass):
        bad: Optional[str] = None

    ExampleModel.model_rebuild()

    onto = owlready2.get_ontology("http://example.org/nonfunc_test#")

    with onto:

        class has_bad(owlready2.DataProperty):  # non-functional on purpose
            range = [str]

        class has_pydowl_type(owlready2.DataProperty, owlready2.FunctionalProperty):
            range = [str]

        class Example(owlready2.Thing):
            pass

    ind = onto.Example("e1")
    ind.has_bad = ["oops"]  # non-functional => list

    ind.has_pydowl_type = f"{ExampleModel.__module__}:{ExampleModel.__qualname__}"

    with pytest.raises(KgPullError):
        ExampleModel.pull_owlready(onto, ind)


# ──────────────────────────────────────────────────────────────────────
# New: KgPushError when dynamic_tbox is disabled
# ──────────────────────────────────────────────────────────────────────


def test_push_owlready_raises_when_dynamic_tbox_disabled(test_onto):
    """
    When dynamic_tbox is False and the OWL class does not exist,
    push_owlready should raise KgPushError.
    """

    class Foo(PydOwlClass):
        x: Optional[int] = None

    foo = Foo(identifier="foo1", x=1)

    with pytest.raises(KgPushError):
        foo.push_owlready(test_onto, dynamic_tbox=False)


# ──────────────────────────────────────────────────────────────────────
# Plant schema + roundtrip tests (from old test_owlready.py)
# ──────────────────────────────────────────────────────────────────────


def test_push1(plant_schema_entities, test_onto):
    """
    push all entities and ensure has_pydowl_type is set
    """
    for e in plant_schema_entities.values():
        e_owl = e.push_owlready(test_onto, dynamic_tbox=True)
        assert (
            e_owl.has_pydowl_type
            == f"{e.__class__.__module__}:{e.__class__.__qualname__}"
        )
    test_onto.save(file="test.owl", format="rdfxml")


def test_push2(plant_schema_entities, test_onto):
    tree1 = plant_schema_entities["tree1"]
    tree2_extract = plant_schema_entities["tree2_extract"]

    tree2_extract.push_owlready(test_onto, dynamic_tbox=True)
    tree2_extract.components = []
    tree2_extract_owl = tree2_extract.push_owlready(test_onto, dynamic_tbox=True)

    # for a non-functional object property, push would not remove triples
    assert tree2_extract_owl.has_components

    # for a functional object property, push will ignore None value
    tree2_extract.plant_origin = None
    tree2_extract_owl = tree2_extract.push_owlready(test_onto, dynamic_tbox=True)
    assert tree2_extract_owl.has_plant_origin

    # for a functional object property, push will UPDATE if the value is not None
    tree2_extract.plant_origin = tree1
    tree2_extract_owl = tree2_extract.push_owlready(test_onto, dynamic_tbox=True)
    assert (
        tree2_extract_owl.has_plant_origin.iri == test_onto.base_iri + tree1.identifier
    )

    test_onto.save(file="test.owl", format="rdfxml")


def test_push3(plant_schema_entities, test_onto):
    tree1 = plant_schema_entities["tree1"]
    setattr(tree1, "field_that_should_not_be_here", "blabla")
    with pytest.raises(ValueError):
        tree1.push_owlready(test_onto, dynamic_tbox=True)


def test_round_trip(plant_schema_entities, test_onto):
    for e in plant_schema_entities.values():
        e_owl = e.push_owlready(test_onto, dynamic_tbox=True)
        e_rec = PydOwlClass.pull_owlready(test_onto, e_owl)
        assert e_rec.model_dump_json() == e.model_dump_json()

    test_onto.save(file="test.owl", format="rdfxml")


# ──────────────────────────────────────────────────────────────────────
# Integration tests (I1/I2/I3) – OWL + Mongo + Pydantic
# ──────────────────────────────────────────────────────────────────────


def test_end_to_end_frontend_to_mongo_to_owl():
    """
    I1: Simulate a frontend payload being validated by Pydantic,
    stored into Mongo-style docs, then pushed into an OWL ontology.
    """

    class Address(PydOwlClass):
        city: Optional[str] = None

    class Person(PydOwlClass):
        name: Optional[str] = None
        address: Optional[Address] = None

    payload = {
        "identifier": "alice",
        "name": "Alice",
        "address": {"identifier": "addr1", "city": "London"},
    }

    alice = Person.model_validate(payload)
    assert isinstance(alice.address, Address)
    assert alice.address.city == "London"

    docs = alice.to_mongo_docs()
    ids = sorted(d["_id"] for d in docs)
    assert ids == ["addr1", "alice"]

    from pydowl.base_class import PydOwlRegistry

    PydOwlRegistry.clear()
    alice2 = Person.from_mongo_docs("alice", docs)
    assert alice2.identifier == "alice"
    assert alice2.address.city == "London"

    onto = owlready2.get_ontology("http://example.org/integration_test#")
    alice2.push_owlready(onto, dynamic_tbox=True)  # alice_ind
    alice_iri = f"{onto.base_iri}alice"
    addr_iri = f"{onto.base_iri}addr1"
    alice_ind2 = onto.search_one(iri=alice_iri)
    addr_ind2 = onto.search_one(iri=addr_iri)
    assert alice_ind2 is not None
    assert addr_ind2 is not None
    assert alice_ind2.has_name == "Alice"
    assert alice_ind2.has_address is addr_ind2
    assert addr_ind2.has_city == "London"


def test_end_to_end_owl_to_pydantic_to_mongo():
    """
    I2: Start from an OWL ontology with individuals, pull into Pydantic
    and then project to Mongo-style docs.
    """

    class Address(PydOwlClass):
        city: Optional[str] = None

    class Person(PydOwlClass):
        name: Optional[str] = None
        address: Optional[Address] = None

    onto = owlready2.get_ontology("http://example.org/integration_test2#")

    with onto:

        class has_name(owlready2.DataProperty, owlready2.FunctionalProperty):
            range = [str]

        class has_city(owlready2.DataProperty, owlready2.FunctionalProperty):
            range = [str]

        class has_address(owlready2.ObjectProperty, owlready2.FunctionalProperty):
            pass

        class has_pydowl_type(owlready2.DataProperty, owlready2.FunctionalProperty):
            range = [str]

        class PersonCls(owlready2.Thing):
            pass

        class AddressCls(owlready2.Thing):
            pass

    addr_ind = onto.AddressCls("addr1")
    addr_ind.has_city = "Paris"

    alice_ind = onto.PersonCls("alice")
    alice_ind.has_name = "Alice"
    alice_ind.has_address = addr_ind

    alice_ind.has_pydowl_type = f"{Person.__module__}:{Person.__qualname__}"
    addr_ind.has_pydowl_type = f"{Address.__module__}:{Address.__qualname__}"

    alice = Person.pull_owlready(onto, alice_ind)
    assert alice.identifier == "alice"
    assert alice.address.city == "Paris"

    docs = alice.to_mongo_docs()
    ids = sorted(d["_id"] for d in docs)
    assert ids == ["addr1", "alice"]

    alice_doc = next(d for d in docs if d["_id"] == "alice")
    addr_doc = next(d for d in docs if d["_id"] == "addr1")

    assert alice_doc["name"] == "Alice"
    assert alice_doc["address"] == "addr1"
    assert addr_doc["city"] == "Paris"


def test_cycle_end_to_end_mongo_and_owl():
    """
    I3: Ensure a cyclic object graph can be serialised to Mongo docs,
    reconstructed from Mongo docs, and pushed to OWL without infinite
    recursion.
    """

    class Person(PydOwlClass):
        name: Optional[str] = None
        spouse: Optional["Person"] = None

    Person.model_rebuild()

    alice = Person(identifier="alice", name="Alice", spouse=None)
    bob = Person(identifier="bob", name="Bob", spouse=None)
    alice.spouse = bob
    bob.spouse = alice

    docs = alice.to_mongo_docs()
    ids = sorted(d["_id"] for d in docs)
    assert ids == ["alice", "bob"]

    from pydowl.base_class import PydOwlRegistry

    PydOwlRegistry.clear()
    alice2 = Person.from_mongo_docs("alice", docs)
    assert alice2.identifier == "alice"
    bob2 = alice2.spouse
    assert bob2.identifier == "bob"
    assert bob2.spouse is alice2

    onto = owlready2.get_ontology("http://example.org/cycle_integration#")
    alice2.push_owlready(onto, dynamic_tbox=True)  # alice_ind
    alice_iri = f"{onto.base_iri}alice"
    bob_iri = f"{onto.base_iri}bob"
    alice_ind2 = onto.search_one(iri=alice_iri)
    bob_ind2 = onto.search_one(iri=bob_iri)
    assert alice_ind2 is not None
    assert bob_ind2 is not None
    assert alice_ind2.has_spouse is bob_ind2
    assert bob_ind2.has_spouse is alice_ind2
