import os
import pytest
from pydowl.sparql.endpoint import SparqlEndpoint


@pytest.fixture(scope="module")
def sparql_endpoint():
    return SparqlEndpoint(
        url=os.getenv("SPARQL_AUTH_ENDPOINT_URL"),
        username=os.getenv("SPARQL_AUTH_ENDPOINT_USER"),
        password=os.getenv("SPARQL_AUTH_ENDPOINT_PASSWORD"),
    )


@pytest.fixture(autouse=True)
def clear_graph(sparql_endpoint, test_onto):
    iri = test_onto.base_iri
    sparql_endpoint.query(f"CLEAR GRAPH <{iri}>")
    yield
    sparql_endpoint.query(f"CLEAR GRAPH <{iri}>")
