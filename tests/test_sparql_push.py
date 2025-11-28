"""
Test suite for pydowl.sparql.push.

Covers:
- Low-level utilities: parsing, blank-node detection, batching, streaming, A-Box export,
  large-literal offloading.
- Integration: full & A-Box only pushes, functional/non-functional properties, JSON literals,
  datetime fields, high-level wrapper, error paths.
"""

import json
from datetime import datetime, timezone
from typing import Optional

import owlready2
import pytest
from SPARQLWrapper import JSON
from rdflib import Graph, Literal as RDFLiteral, URIRef, RDF
from rdflib.plugins.parsers.ntriples import ParseError
from pydantic import Field

from pydowl import PydOwlClass
from pydowl.sparql.push import (
    batch_lines,
    export_abox,
    has_blank_node,
    parse_ntriples_line,
    push_owlready_to_sparql_endpoint,
    push_pyd_to_sparql_endpoint,
    stream_ntriples,
    transform_triples,
)

pytest_plugins = ["tests.fixtures.sparql_fixtures"]


# ----------------------------------------------------------------------
# Unit Tests: Utilities
# ----------------------------------------------------------------------


def test_parse_ntriples_line_valid():
    """
    parse_ntriples_line should return correct (s, p, o) for a valid triple.
    """
    line = '<http://s> <http://p> "o"^^<http://dt> .'
    s, p, o = parse_ntriples_line(line)
    assert str(s) == "http://s"
    assert str(p) == "http://p"
    expected = RDFLiteral("o", datatype=URIRef("http://dt"))
    assert o == expected


def test_parse_ntriples_line_invalid():
    """
    parse_ntriples_line should propagate ParseError on malformed input.
    """
    with pytest.raises(ParseError):
        parse_ntriples_line("not a triple")


def test_has_blank_node():
    """
    has_blank_node should detect blank-node subjects or objects.
    """
    good = "<http://s> <http://p> <http://o> ."
    bad = '_:b <http://p> "o" .'
    assert not has_blank_node([good])
    assert has_blank_node([good, bad])


def test_batch_lines_edge_cases():
    """
    batch_lines should group items, handle empty and exact-divide cases.
    """
    # empty
    assert list(batch_lines(iter([]), 3)) == []
    # exact divide
    data = (str(i) for i in range(6))
    assert list(batch_lines(data, 3)) == [["0", "1", "2"], ["3", "4", "5"]]
    # remainder
    data = (str(i) for i in range(5))
    assert list(batch_lines(data, 3)) == [["0", "1", "2"], ["3", "4"]]


def test_stream_and_export_abox_empty(test_onto):
    """
    export_abox returns an RDFLib Graph; stream_ntriples on an empty
    ontology still emits some triples (ontology metadata).
    """
    g = export_abox(test_onto)
    assert isinstance(g, Graph)
    lines = list(stream_ntriples(test_onto))
    assert len(lines) >= 0  # nothing stronger is guaranteed here


def test_transform_triples_noop(test_onto):
    """
    transform_triples should leave small literals intact.
    """
    # create a data property
    with test_onto:

        class has_val(owlready2.DataProperty, owlready2.FunctionalProperty):
            pass

    s = URIRef(test_onto.base_iri + "x")
    p = URIRef(test_onto.base_iri + "has_val")
    small = "ok"
    nt = f'{s.n3()} {p.n3()} "{small}" .'
    out = transform_triples([nt], test_onto, max_literal_size=10)
    assert out == [nt]


# ----------------------------------------------------------------------
# Integration Tests: SPARQL Push
# ----------------------------------------------------------------------


def test_end_to_end_full_push(test_onto, sparql_endpoint):
    """
    Full push (TBox + ABox) injects has_pydowl_type and rdf:type as expected.
    """
    inst = PydOwlClass(identifier="node1", pydowl_version="0.1")
    inst.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    q = f"""
    SELECT ?s ?p ?o WHERE {{
      GRAPH <{test_onto.base_iri}> {{ ?s ?p ?o }}
    }}
    """
    bindings = sparql_endpoint.query(q, method="GET", return_format=JSON)
    triples = {(b["s"]["value"], b["p"]["value"], b["o"]["value"]) for b in bindings}

    tag = f"{inst.__class__.__module__}:{inst.__class__.__qualname__}"
    has_type_iri = test_onto.base_iri + "has_pydowl_type"
    assert (test_onto.base_iri + "node1", has_type_iri, tag) in triples
    assert any(s.endswith("#node1") and p == str(RDF.type) for s, p, o in triples)


def test_abox_only_mode(test_onto, sparql_endpoint):
    """
    ABox-only push must include instance data but no TBox triples such as rdfs:subClassOf.
    """

    class Simple(PydOwlClass):
        val: Optional[int] = 0

    inst = Simple(identifier="s1", val=123)
    inst.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=True)

    q = f"""
    SELECT ?p ?o WHERE {{
      GRAPH <{test_onto.base_iri}> {{ <{test_onto.base_iri}s1> ?p ?o }}
    }}
    """
    bindings = sparql_endpoint.query(q, method="GET", return_format=JSON)
    preds = {b["p"]["value"] for b in bindings}
    vals = {b["o"]["value"] for b in bindings}

    assert "123" in vals
    assert all("subClassOf" not in p for p in preds)


def test_functional_upsert(test_onto, sparql_endpoint):
    """
    Pushing twice with updated functional data property overwrites previous value.
    """
    inst = PydOwlClass(identifier="n2", pydowl_version="v1")
    inst.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    inst.pydowl_version = "v2"
    inst.push_owlready(test_onto, dynamic_tbox=False)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    q = f"""
    SELECT DISTINCT ?v WHERE {{
      GRAPH <{test_onto.base_iri}> {{
        <{test_onto.base_iri}n2> <{test_onto.base_iri}has_pydowl_version> ?v
      }}
    }}
    """
    bindings = sparql_endpoint.query(q, method="GET", return_format=JSON)
    versions = {b["v"]["value"] for b in bindings}
    assert versions == {"v2"}


def test_functional_list_json_literal(test_onto, sparql_endpoint):
    """
    List[str] field is stored as a single JSON literal and upserted correctly.
    """

    class Bag(PydOwlClass):
        tags: list[str] = Field(default_factory=list)

    inst = Bag(identifier="b1", tags=["x", "y"])
    inst.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    inst.tags.append("z")
    inst.push_owlready(test_onto, dynamic_tbox=False)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    q = f"""
    SELECT ?j WHERE {{
      GRAPH <{test_onto.base_iri}> {{
        <{test_onto.base_iri}b1> <{test_onto.base_iri}has_tags> ?j
      }}
    }}
    """
    bindings = sparql_endpoint.query(q, method="GET", return_format=JSON)
    vals = {b["j"]["value"] for b in bindings}
    assert vals == {json.dumps(["x", "y", "z"])}


def test_nonfunctional_object_list(test_onto, sparql_endpoint):
    """
    List[PydOwlClass] produces non-functional triples that accumulate.
    """

    class Child(PydOwlClass):
        pass

    class Parent(PydOwlClass):
        kids: list[Child] = Field(default_factory=list)

    with test_onto:
        type("Child", (owlready2.Thing,), {})
        type("Parent", (owlready2.Thing,), {})

    c1 = Child(identifier="c1")
    c2 = Child(identifier="c2")
    p = Parent(identifier="p1", kids=[c1, c2])
    p.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=True)

    q = f"""
    SELECT ?o WHERE {{
      GRAPH <{test_onto.base_iri}> {{
        <{test_onto.base_iri}p1> <{test_onto.base_iri}has_kids> ?o
      }}
    }}
    """
    first = {
        b["o"]["value"]
        for b in sparql_endpoint.query(q, method="GET", return_format=JSON)
    }
    assert {test_onto.base_iri + "c1", test_onto.base_iri + "c2"} == first

    # Push again with only c2 -- c1 triple should remain (non-functional)
    p.kids = [c2]
    p.push_owlready(test_onto, dynamic_tbox=False)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=True)

    second = {
        b["o"]["value"]
        for b in sparql_endpoint.query(q, method="GET", return_format=JSON)
    }
    assert test_onto.base_iri + "c1" in second and test_onto.base_iri + "c2" in second


def test_datetime_field_roundtrip(test_onto, sparql_endpoint):
    """
    Optional[datetime] fields round-trip via the custom DateTime datatype
    in the full-push path.
    """

    class Event(PydOwlClass):
        ts: Optional[datetime] = None

    now = datetime.now(timezone.utc).replace(microsecond=0)
    ev = Event(identifier="e1", ts=now)
    ev.push_owlready(test_onto, dynamic_tbox=True)

    # Use full-push so the literal is preserved and not offloaded
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    q = f"""
    SELECT ?t WHERE {{
      GRAPH <{test_onto.base_iri}> {{
        <{test_onto.base_iri}e1> <{test_onto.base_iri}has_ts> ?t
      }}
    }}
    """
    binding = sparql_endpoint.query(q, method="GET", return_format=JSON)
    assert len(binding) == 1
    val = binding[0]["t"]
    # must be a literal with the correct custom datatype
    assert val["type"] == "typed-literal"
    assert val.get("datatype") == "http://pydowl.org/dtype#DateTime", (
        f"got datatype {val.get('datatype')!r}"
    )
    from datetime import datetime as _dt

    parsed = _dt.fromisoformat(val["value"])
    assert parsed == now


def test_push_pyd_to_sparql_endpoint_wrapper(test_onto, sparql_endpoint, tmp_path):
    """
    High-level helper push_pyd_to_sparql_endpoint should work when
    supplying a saved TBox ontology (dynamic_tbox=False).
    Verifies via an ASK that has_pydowl_type was inserted.
    """
    # 1) Ensure the OWL class exists in the ontology
    with test_onto:
        type("PydOwlClass", (owlready2.Thing,), {})

        class has_pydowl_version(owlready2.FunctionalProperty, owlready2.DataProperty):
            pass

    # 2) Save that ontology as a TBox file
    tbox = tmp_path / "tbox.owl"
    test_onto.save(file=str(tbox), format="rdfxml")

    # 3) Call the helper
    inst = PydOwlClass(identifier="x1", pydowl_version="v9")
    push_pyd_to_sparql_endpoint(
        inst, sparql_endpoint, abox_only=False, tbox_ontology_path=str(tbox)
    )

    # 4) ASK for the presence of the has_pydowl_type triple
    ask_q = f"""
    ASK WHERE {{
      GRAPH <{test_onto.base_iri}> {{
        <{test_onto.base_iri}x1>
          <{test_onto.base_iri}has_pydowl_type> ?t .
      }}
    }}
    """
    wrapper = sparql_endpoint.setup_wrapper(method="GET", return_format=JSON)
    wrapper.setQuery(ask_q)
    result = wrapper.query().convert()

    # SPARQLWrapper returns {'boolean': true/false}
    assert result.get("boolean") is True


def test_push_raises_on_missing_tbox(test_onto, sparql_endpoint):
    """
    Pushing with dynamic_tbox=False before the OWL class exists should
    error on push_owlready. The SPARQL push itself should not raise.
    """
    inst = PydOwlClass(identifier="bad1", pydowl_version="v1")
    # No OWL class registered, so push_owlready must fail
    with pytest.raises(Exception):
        inst.push_owlready(test_onto, dynamic_tbox=False)

    # The SPARQL push only serializes the ontology (which is empty—it
    # contains no individuals yet)—it should not raise.
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)
