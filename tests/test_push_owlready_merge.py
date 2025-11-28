import owlready2
import pytest
from typing import List, Optional

from pydantic import Field

from pydowl.base_class import PydOwlClass, PydOwlRegistry


class Child(PydOwlClass):
    name: Optional[str] = None


class Parent(PydOwlClass):
    name: Optional[str] = None
    child_opt: Optional[Child] = None
    children: List[Child] = Field(default_factory=list)


@pytest.fixture(autouse=True)
def clear_registry():
    """
    Ensure the global PydOwlRegistry does not leak state between tests.
    """
    PydOwlRegistry.clear()
    yield
    PydOwlRegistry.clear()


def make_ontology(
    iri: str = "http://example.org/test_push_merge#",
) -> owlready2.Ontology:
    world = owlready2.World()
    onto = world.get_ontology(iri)
    return onto


def _get_parent_and_children(onto: owlready2.Ontology):
    base = onto.base_iri
    parent_iri = f"{base}p"
    parent_ind = onto.search_one(iri=parent_iri)
    assert parent_ind is not None

    has_children_prop = onto.search_one(iri=f"{base}has_children")
    assert has_children_prop is not None

    children_inds = list(has_children_prop[parent_ind])
    return parent_ind, children_inds


def test_push_owlready_merge_true_accumulates_list_values():
    """
    With merge=True (default), repeated pushes accumulate list values
    in the OWL graph (union-sert semantics).
    """
    onto = make_ontology()

    c1 = Child(identifier="c1", name="c1")
    c2 = Child(identifier="c2", name="c2")
    p = Parent(identifier="p", name="parent", children=[c1])

    # First push: only c1 is present.
    p.push_owlready(onto, dynamic_tbox=True)  # merge=True by default

    # Second push: children changed to [c2]; merge=True should retain c1
    # and append c2 in the OWL graph.
    p.children = [c2]
    p.push_owlready(onto, dynamic_tbox=True)  # merge=True

    _, children_inds = _get_parent_and_children(onto)
    iri_set = {ind.iri for ind in children_inds}

    base = onto.base_iri
    assert f"{base}c1" in iri_set
    assert f"{base}c2" in iri_set


def test_push_owlready_merge_false_replaces_list_values():
    """
    With merge=False, repeated pushes replace list values so the OWL
    graph matches the current Python list exactly.
    """
    onto = make_ontology()

    c1 = Child(identifier="c1", name="c1")
    c2 = Child(identifier="c2", name="c2")
    p = Parent(identifier="p", name="parent", children=[c1])

    # First push with merge=False: OWL has only c1.
    p.push_owlready(onto, dynamic_tbox=True, merge=False)

    # Change children to [c2] and push again with merge=False: OWL
    # should now only reference c2.
    p.children = [c2]
    p.push_owlready(onto, dynamic_tbox=True, merge=False)

    _, children_inds = _get_parent_and_children(onto)
    iri_set = {ind.iri for ind in children_inds}

    base = onto.base_iri
    assert iri_set == {f"{base}c2"}


def test_push_owlready_merge_false_clears_list_when_empty():
    """
    merge=False with an empty list should clear any previous OWL values
    for that list property.
    """
    onto = make_ontology()

    c1 = Child(identifier="c1", name="c1")
    p = Parent(identifier="p", name="parent", children=[c1])

    # First push: children = [c1].
    p.push_owlready(onto, dynamic_tbox=True, merge=False)

    # Now clear the list and push again.
    p.children = []
    p.push_owlready(onto, dynamic_tbox=True, merge=False)

    _, children_inds = _get_parent_and_children(onto)
    assert children_inds == []


@pytest.mark.parametrize("merge", [True, False])
def test_push_owlready_merge_flag_does_not_affect_optional_relationship(merge: bool):
    """
    The merge flag only affects LIST_PYD_CLS fields; OPTIONAL_PYD_CLS
    fields should always behave as replace semantics.
    """
    onto = make_ontology()

    c1 = Child(identifier="c1", name="c1")
    c2 = Child(identifier="c2", name="c2")
    p = Parent(identifier="p", name="parent", child_opt=c1)

    # First push: child_opt = c1.
    p.push_owlready(onto, dynamic_tbox=True, merge=merge)

    # Second push: child_opt = c2 should overwrite the previous value
    # regardless of merge flag.
    p.child_opt = c2
    p.push_owlready(onto, dynamic_tbox=True, merge=merge)

    base = onto.base_iri
    parent_iri = f"{base}p"
    parent_ind = onto.search_one(iri=parent_iri)
    assert parent_ind is not None

    has_child_opt = onto.search_one(iri=f"{base}has_child_opt")
    assert has_child_opt is not None

    values = list(has_child_opt[parent_ind])
    assert len(values) == 1
    assert values[0].iri == f"{base}c2"


def test_to_mongo_docs_and_from_mongo_docs_retain_replace_semantics_for_lists():
    """
    Verify that Mongo serialisation/deserialisation continues to use
    replace (not union) semantics for list fields, independent of the
    merge flag in push_owlready.
    """
    # Construct a small graph in memory.
    c1 = Child(identifier="c1", name="c1")
    c2 = Child(identifier="c2", name="c2")
    p = Parent(identifier="p", name="parent", children=[c1])

    # First projection to Mongo docs.
    docs1 = p.to_mongo_docs()
    parent_doc1 = next(doc for doc in docs1 if doc["_id"] == "p")
    assert parent_doc1["children"] == ["c1"]

    # Change children to [c2] and project again; we expect replacement,
    # not union at the document level.
    p.children = [c2]
    docs2 = p.to_mongo_docs()
    parent_doc2 = next(doc for doc in docs2 if doc["_id"] == "p")
    assert parent_doc2["children"] == ["c2"]

    # Now reconstruct from docs2 and ensure the Python object graph
    # also reflects [c2] only.
    PydOwlRegistry.clear()
    root = Parent.from_mongo_docs("p", docs2)
    assert isinstance(root, Parent)
    assert [child.identifier for child in root.children] == ["c2"]
