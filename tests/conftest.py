import owlready2
import pytest

try:
    from dotenv import load_dotenv
    from pathlib import Path

    # Load test-time environment variables (e.g. SPARQL endpoint config)
    load_dotenv(Path(__file__).with_name(".env"), override=True)
except Exception:
    # It is safe to run tests without a .env; SPARQL tests will skip
    # themselves automatically if required env vars are missing.
    pass

# Register shared fixtures from the `tests/fixtures` package.
# - plant_schema: domain-specific PydOwlClass instances for OWL tests.
# - sparql_fixtures: SPARQL endpoint + graph-clearing fixtures.
pytest_plugins = [
    "tests.fixtures.plant_schema",
    "tests.fixtures.sparql_fixtures",
]


@pytest.fixture(scope="function")
def test_onto():
    """
    Provide a fresh Owlready2 ontology per test, backed by the global
    default world. The base IRI is fixed, and the ontology plus its
    entities are fully removed after the test.
    """
    iri = "http://example.org/test.owl#"  # fixed IRI is OK for tests
    onto = owlready2.get_ontology(iri)
    yield onto

    # Clear individuals, classes, and properties
    for ind in list(onto.individuals()):
        owlready2.destroy_entity(ind)
    for cls in list(onto.classes()):
        owlready2.destroy_entity(cls)
    for prop in list(onto.properties()):
        owlready2.destroy_entity(prop)
    onto.destroy(update_relation=True, update_is_a=True)
    # Remove namespace from default world to avoid leaks between tests
    owlready2.default_world._namespaces.pop(iri[:-1], None)


@pytest.fixture(scope="function", autouse=True)
def clear_registry():
    """
    Ensure PydOwlRegistry is empty between tests, so object identity
    behaviour is deterministic and tests cannot interfere with one
    another via the global registry.
    """
    from pydowl.base_class import PydOwlRegistry

    PydOwlRegistry.clear()
    yield
    PydOwlRegistry.clear()
