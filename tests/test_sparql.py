from __future__ import annotations

import os
from typing import Optional

import pytest

from pydowl import PydOwlClass
from pydowl.sparql import (
    push_owlready_to_sparql,
    push_pyd_to_sparql_endpoint,
    pull_owlready_from_sparql_endpoint,
    pull_pyd_from_sparql_endpoint,
)
from pydowl.sparql.settings import PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI
from pydowl.sparql.endpoint import SparqlEndpoint

SPARQL_URL = os.getenv("SPARQL_AUTH_ENDPOINT_URL")
requires_sparql = pytest.mark.skipif(
    not SPARQL_URL, reason="SPARQL_AUTH_ENDPOINT_URL is not configured"
)


@requires_sparql
def test_push_owlready_and_pull_owlready_roundtrip(
    test_onto, sparql_endpoint: SparqlEndpoint
):
    """
    Push an Owlready2 ontology to a Virtuoso/GraphDB SPARQL endpoint and
    pull it back into a new Owlready2 ontology, verifying basic triples.
    """

    class Person(PydOwlClass):
        name: Optional[str] = None
        age: Optional[int] = None

    # Build ontology with one individual
    p = Person(identifier="alice", name="Alice", age=42)
    p.push_owlready(test_onto, dynamic_tbox=True)

    iri = test_onto.base_iri

    # Push full ontology
    push_owlready_to_sparql(test_onto, sparql_endpoint, graph_iri=iri, abox_only=False)

    # Check triples exist via SELECT
    rows = sparql_endpoint.query_select(
        f"SELECT ?s ?p ?o WHERE {{ GRAPH <{iri}> {{ ?s ?p ?o }} }}"
    )
    s_iri = f"{iri}{p.identifier}"
    has_name_iri = f"{iri}has_name"
    assert any(
        r["s"]["value"] == s_iri and r["p"]["value"] == has_name_iri for r in rows
    )

    # Pull ontology back
    onto2 = pull_owlready_from_sparql_endpoint(iri, sparql_endpoint)
    ind2 = onto2.search_one(iri=s_iri)
    assert ind2 is not None
    assert hasattr(ind2, "has_name")
    assert ind2.has_name == "Alice"


@requires_sparql
def test_push_pyd_and_pull_pyd_roundtrip(sparql_endpoint: SparqlEndpoint):
    """
    Use push_pyd_to_sparql_endpoint (temporary ontology) to push a
    PydOwlClass, then pull it back with pull_pyd_from_sparql_endpoint.
    """

    class Person(PydOwlClass):
        name: Optional[str] = None
        age: Optional[int] = None

    # Clear the temporary graph before test (if it exists)
    temp_graph = PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI
    sparql_endpoint.query(f"CLEAR GRAPH <{temp_graph}>")

    p = Person(identifier="bob", name="Bob", age=33)

    # Push via temporary ontology (dynamic_tbox=True branch)
    push_pyd_to_sparql_endpoint(p, sparql_endpoint, abox_only=False)

    # Pull back as PydOwlClass
    ind_iri = f"{temp_graph}{p.identifier}"
    p2 = pull_pyd_from_sparql_endpoint(ind_iri, temp_graph, sparql_endpoint)

    assert isinstance(p2, Person)
    assert p2.identifier == p.identifier
    assert p2.name == "Bob"
    assert p2.age == 33

    p3 = pull_pyd_from_sparql_endpoint(
        ind_iri, temp_graph, sparql_endpoint, mode="abox"
    )
    assert isinstance(p3, Person)
    assert p3.name == "Bob"
    assert p3.age == 33

    # Clear temporary graph after test
    sparql_endpoint.query(f"CLEAR GRAPH <{temp_graph}>")
