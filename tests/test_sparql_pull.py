from typing import Optional

import owlready2
import pytest
from rdflib.plugins.parsers.ntriples import ParseError

from pydowl import PydOwlClass
from pydowl.base_class import KgPullError
from pydowl.sparql.pull import (
    examine_properties,
    generate_construct_pattern,
    pull_owlready_from_sparql_endpoint,
    pull_pyd_from_sparql_endpoint,
    get_individual_properties,
    get_individual_classes,
    clear_examine_cache,
)
from pydowl.sparql.push import (
    parse_ntriples_line,
    has_blank_node,
    push_owlready_to_sparql_endpoint,
)

pytest_plugins = ["tests.fixtures.sparql_fixtures"]


# -------------------------------------------------------------------
# Unit tests: examine_properties, SPARQL‐SELECT parsers
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def flush_pull_cache():
    clear_examine_cache()
    yield
    clear_examine_cache()


def test_examine_properties_empty_ontology(test_onto):
    # No properties defined → lists are empty
    data_props, func_objs, nonfunc_objs, dt_ranges = examine_properties(test_onto)
    assert data_props == []
    assert func_objs == []
    assert nonfunc_objs == []
    assert dt_ranges == {}


def test_examine_properties_with_props(test_onto):
    # Dynamically add a DataProperty and ObjectProperty
    with test_onto:

        class DataPropTestDynamic(owlready2.DataProperty, owlready2.FunctionalProperty):
            pass

        class ObjectPropTestDynamic(owlready2.ObjectProperty):
            pass

    data_props, func_objs, nonfunc_objs, dt_ranges = examine_properties(test_onto)
    assert any(p.endswith("#DataPropTestDynamic") for p in data_props)
    assert dt_ranges  # should include D’s range
    assert any(p.endswith("#ObjectPropTestDynamic") for p in nonfunc_objs)


def test_get_individual_classes_and_properties_unit():
    # Fake SPARQL JSON responses
    fake_classes = {
        "results": {
            "bindings": [
                {"class": {"value": "http://ex.org#A"}, "kind": {"value": "Leaf"}},
                {"class": {"value": "http://ex.org#B"}, "kind": {"value": "Super"}},
            ]
        }
    }
    fake_props = {
        "results": {
            "bindings": [
                {
                    "p": {"value": "http://ex.org#p1"},
                    "o": {"value": "42"},
                    "dt": {"value": "http://www.w3.org/2001/XMLSchema#integer"},
                    "pt": {"value": "data"},
                },
                {
                    "p": {"value": "http://ex.org#p2"},
                    "o": {"value": "http://ex.org#i1"},
                    "dt": {"value": None},
                    "pt": {"value": "object"},
                },
            ]
        }
    }

    class DummyEP:
        def __init__(self, data):
            self.data = data

        def query_select(self, q: str):
            # Ignore the query; just return bindings
            return self.data["results"]["bindings"]

    ep1 = DummyEP(fake_classes)
    ep2 = DummyEP(fake_props)

    cls_leaf = get_individual_classes("i", "g", ep1, leaf_only=True)
    assert cls_leaf == ["http://ex.org#A"]

    props = get_individual_properties("i", "g", ep2)
    assert "http://ex.org#p1" in props
    assert props["http://ex.org#p1"][0][:2] == ("42", "data")
    assert props["http://ex.org#p2"][0][1] == "object"


# -------------------------------------------------------------------
# Integration: CONSTRUCT → Owlready2 → SPARQL → Owlready2 roundtrip
# -------------------------------------------------------------------


def test_pull_owlready_roundtrip(test_onto, sparql_endpoint):
    # 1) Create an individual in test_onto
    with test_onto:
        C = type("C1", (owlready2.Thing,), {})
        inst = C(iri=test_onto.base_iri + "x1")

        # add a data property
        class has_D(owlready2.DataProperty, owlready2.FunctionalProperty):
            pass

        inst.has_D = "foo"

    # 2) Push entire ontology to SPARQL
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    # 3) Pull back into a fresh ontology
    onto2 = pull_owlready_from_sparql_endpoint(test_onto.base_iri, sparql_endpoint)
    got = onto2.search_one(iri=test_onto.base_iri + "x1")
    assert got is not None
    assert got.has_D == "foo"


# -------------------------------------------------------------------
# Integration: PydOwlClass → SPARQL → PydOwlClass roundtrip (full mode)
# -------------------------------------------------------------------


def test_pull_pyd_class_roundtrip_simple(test_onto, sparql_endpoint):
    # Define a simple PydOwlClass with an int
    class S(PydOwlClass):
        val: Optional[int] = None

    # 1) Instantiate & push
    s1 = S(identifier="s2", val=123)
    s1.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    # 2) Pull back via full mode
    s2 = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "s2",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
        mode="full",
    )
    assert isinstance(s2, S)
    assert s2.val == 123


def test_pull_pyd_class_roundtrip_nested_full(test_onto, sparql_endpoint):
    # Define nested classes
    class Child(PydOwlClass):
        name: Optional[str] = None

    class Parent(PydOwlClass):
        child: Optional[Child] = None

    # Register OWL classes
    with test_onto:
        type("Child", (owlready2.Thing,), {})
        type("Parent", (owlready2.Thing,), {})

    # Push nested object
    c = Child(identifier="c9", name="Alan")
    p = Parent(identifier="p9", child=c)
    p.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    # Pull back via full mode
    pr = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "p9",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
        mode="full",
    )
    assert isinstance(pr, Parent)
    assert isinstance(pr.child, Child)
    assert pr.child.name == "Alan"


def test_pull_pyd_class_roundtrip_abox_mode(test_onto, sparql_endpoint):
    # Same as above, but use ABox-only mode to exercise Tier 2
    class Child(PydOwlClass):
        name: Optional[str] = None

    class Parent(PydOwlClass):
        child: Optional[Child] = None

    with test_onto:
        type("Child", (owlready2.Thing,), {})
        type("Parent", (owlready2.Thing,), {})

    c = Child(identifier="c10", name="Ada")
    p = Parent(identifier="p10", child=c)
    p.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    pr = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "p10",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
        mode="abox",
        max_depth=2,
    )
    assert isinstance(pr, Parent)
    assert isinstance(pr.child, Child)
    assert pr.child.name == "Ada"


def test_pull_pyd_override_class_full(test_onto, sparql_endpoint):
    # When pyd_cls is passed explicitly, use that
    class A(PydOwlClass):
        x: Optional[int] = None

    # Register class
    with test_onto:
        type("A", (owlready2.Thing,), {})

    a = A(identifier="ov1", x=7)
    a.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    # Pull forcing base PydOwlClass, not A
    base_obj = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "ov1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
        pyd_cls=PydOwlClass,
        mode="full",
    )
    assert isinstance(base_obj, PydOwlClass)
    # It will still have x in extra fields
    assert base_obj.__dict__["x"] == 7


def test_pull_missing_discriminator_raises_kgpullerror(test_onto, sparql_endpoint):
    # Create an individual with no has_pydowl_type
    with test_onto:

        class NoTag(owlready2.Thing):
            pass

        NoTag(iri=test_onto.base_iri + "nt1")
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    with pytest.raises(KgPullError):
        pull_pyd_from_sparql_endpoint(
            ind_iri=test_onto.base_iri + "nt1",
            graph_iri=test_onto.base_iri,
            sparql_ep=sparql_endpoint,
        )


# ----------------------------------------------------------------------
# parse_ntriples_line now raises ParseError for bad input
# ----------------------------------------------------------------------


def test_parse_ntriples_line_malformed_raises_parseerror():
    with pytest.raises(ParseError):
        parse_ntriples_line("this is not even close to valid ntriples")


# ----------------------------------------------------------------------
# has_blank_node should not swallow parse errors
# ----------------------------------------------------------------------


def test_has_blank_node_raises_on_invalid_line():
    bad_line = "<<< bad ntriples >>>"
    with pytest.raises(ParseError):
        # previously would return False; now should propagate
        has_blank_node([bad_line])


# ----------------------------------------------------------------------
# examine_properties cache
# ----------------------------------------------------------------------


def test_examine_properties_caching(test_onto):
    # first call populates cache
    res1 = examine_properties(test_onto)
    # monkey-patch onto.properties to error if called again
    orig = test_onto.properties

    def fail():
        raise RuntimeError("should not re-scan")

    test_onto.properties = fail  # type: ignore
    res2 = examine_properties(test_onto)
    assert res1 is res2
    # restore
    test_onto.properties = orig


# ----------------------------------------------------------------------
# generate_construct_pattern
# ----------------------------------------------------------------------


def test_generate_construct_pattern_simple():
    class M(PydOwlClass):
        a: Optional[int] = None
        b: Optional[str] = None

    pattern = generate_construct_pattern(
        M, "http://ex.org/graph#", "?s", set(), depth=1
    )
    # must reference both properties
    assert "?s <http://ex.org/graph#has_a> ?a ." in pattern
    assert "?s <http://ex.org/graph#has_b> ?b ." in pattern


# ----------------------------------------------------------------------
# pull_owlready_from_sparql_endpoint adds discriminator prop if missing
# ----------------------------------------------------------------------


def test_pull_owlready_adds_has_pydowl_type(test_onto, sparql_endpoint):
    # create a simple individual without pydowl_type
    with test_onto:
        C = type("C2", (owlready2.Thing,), {})
        # create an instance
        C(iri=test_onto.base_iri + "x2")
    # push raw ontology
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    onto2 = pull_owlready_from_sparql_endpoint(test_onto.base_iri, sparql_endpoint)
    # must now have a DataProperty named has_pydowl_type
    names = {p.name for p in onto2.data_properties()}
    assert "has_pydowl_type" in names


# ----------------------------------------------------------------------
# get_individual_properties integration
# ----------------------------------------------------------------------


def test_get_individual_properties_data_and_object(test_onto, sparql_endpoint):
    # Define properties & two individuals
    with test_onto:

        class has_data(owlready2.DataProperty, owlready2.FunctionalProperty):
            pass

        class has_obj(owlready2.ObjectProperty, owlready2.FunctionalProperty):
            pass

        D = type("D3", (owlready2.Thing,), {})

        d1 = D(iri=test_onto.base_iri + "d1")
        d2 = D(iri=test_onto.base_iri + "d2")
        d1.has_data = "foo"
        d1.has_obj = d2

    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    props = get_individual_properties(
        ind_iri=test_onto.base_iri + "d1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
    )
    # two entries: one data, one object
    assert any(
        pt == "data" and lex == "foo"
        for lex, pt, _ in props[test_onto.base_iri + "has_data"]
    )
    assert any(
        pt == "object" and lex.endswith("#d2")
        for lex, pt, _ in props[test_onto.base_iri + "has_obj"]
    )
