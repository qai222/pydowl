"""
pydowl.sparql
=============

Helpers for pushing and pulling :class:`pydowl.base_class.PydOwlClass`
instances to and from SPARQL 1.1 endpoints (e.g. Virtuoso, GraphDB).

Core entry points
-----------------

Push:

* :func:`push_owlready_to_sparql` –
  push an Owlready2 ontology (TBox + ABox or ABox-only) to a named graph.
* :func:`push_pyd_to_sparql_endpoint` –
  push a single :class:`pydowl.base_class.PydOwlClass` instance by
  materialising it into an Owlready2 ontology and then calling the
  above.

Pull:

* :func:`pull_owlready_from_sparql_endpoint` –
  load a named graph into an Owlready2 ontology, resolving large-literal
  placeholders from Azure if configured.
* :func:`pull_pyd_from_sparql_full` –
  "Tier 1" full-graph pull: load the entire named graph into Owlready2
  and delegate to :meth:`pydowl.base_class.PydOwlClass.pull_owlready`.
* :func:`pull_pyd_from_sparql_abox` –
  "Tier 2" ABox-only pull: use schema-driven SPARQL SELECT queries to
  fetch just the facts needed for a given individual (and its
  neighbours), then hydrate Pydantic models directly.
* :func:`pull_pyd_from_sparql_endpoint` –
  unified helper that selects between full-graph and ABox-only pulls
  via the ``mode`` argument (``"full"`` or ``"abox"``).

The module also exposes lower-level utilities (e.g.
:class:`SparqlEndpoint`, Azure helpers, and pattern generators) for
advanced use cases.
"""

from .settings import (
    LARGE_LITERAL_THRESHOLD,
    AZURE_CONTAINER_NAME,
    ENV_AZURE_STORAGE_CONNECTION_STRING,
    ENV_AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN,
    LARGE_NODE_IRI_SUFFIX,
    PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI,
)
from .utils_azure import (
    blob_upload_string_or_bytes,
    blob_download_bytes_or_str,
)
from .endpoint import SparqlEndpoint
from .push import (
    PushError,
    stream_ntriples,
    export_abox,
    parse_ntriples_line,
    has_blank_node,
    batch_lines,
    transform_triples,
    upload_large_node,
    push_owlready_to_sparql,
    push_owlready_to_sparql_endpoint,  # backward-compat alias
    push_pyd_to_sparql_endpoint,
)
from .pull import (
    examine_properties,
    clear_examine_cache,
    pull_owlready_from_sparql_endpoint,
    pull_pyd_from_sparql_full,
    pull_pyd_from_sparql_abox,
    pull_pyd_from_sparql_endpoint,
    get_individual_properties,
    get_individual_classes,
    generate_construct_pattern,
)

__all__ = [
    # settings
    "LARGE_LITERAL_THRESHOLD",
    "AZURE_CONTAINER_NAME",
    "ENV_AZURE_STORAGE_CONNECTION_STRING",
    "ENV_AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN",
    "LARGE_NODE_IRI_SUFFIX",
    "PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI",
    # utils
    "blob_upload_string_or_bytes",
    "blob_download_bytes_or_str",
    # endpoint
    "SparqlEndpoint",
    # push
    "PushError",
    "stream_ntriples",
    "export_abox",
    "parse_ntriples_line",
    "has_blank_node",
    "batch_lines",
    "transform_triples",
    "upload_large_node",
    "push_owlready_to_sparql",
    "push_owlready_to_sparql_endpoint",
    "push_pyd_to_sparql_endpoint",
    # pull
    "examine_properties",
    "clear_examine_cache",
    "pull_owlready_from_sparql_endpoint",
    "pull_pyd_from_sparql_full",
    "pull_pyd_from_sparql_abox",
    "pull_pyd_from_sparql_endpoint",
    "get_individual_properties",
    "get_individual_classes",
    "generate_construct_pattern",
]
