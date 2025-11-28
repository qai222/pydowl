"""
pydowl.sparql.push
==================

Streaming and pushing OWL ABox or full ontology data via Owlready2 to a
SPARQL 1.1 endpoint.

Features
--------
* Memory-efficient streaming of N-Triples from Owlready2
* Optional ABox-only export
* Blank-node detection (fail-fast)
* Large-literal offloading to Azure Blob Storage
* Chunked SPARQL updates with functional-property upserts
"""

import hashlib
import io
import json
import os
import uuid
from typing import Generator, List, Union, Optional

from loguru import logger
from owlready2 import Ontology, FunctionalProperty, World, destroy_entity, Thing
from rdflib import Graph, Namespace, URIRef, Literal, BNode, RDF
from rdflib.plugins.parsers.ntriples import ParseError

from pydowl.base_class import PydOwlClass
from .endpoint import SparqlEndpoint
from .settings import (
    ENV_AZURE_STORAGE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
    LARGE_LITERAL_THRESHOLD,
    LARGE_NODE_IRI_SUFFIX,
    PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI,
)
from .utils_azure import blob_upload_string_or_bytes

FilePath = Union[str, bytes]


def upload_large_node(
    large_node: str | bytes,
    storage_file_basename: str,
    storage_option: str = "azure",
) -> str:
    """
    Upload a large literal payload to external storage and return a URL.

    Currently only Azure Blob Storage is supported; other storage
    backends would require extending this function.
    """
    logger.debug(f"uploading to: {storage_option}")
    logger.debug(f"uploading file basename: {storage_file_basename}")
    if storage_option == "azure":
        conn_str = os.getenv(ENV_AZURE_STORAGE_CONNECTION_STRING)
        if not conn_str:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")
        container_name = AZURE_CONTAINER_NAME
        upload_instance = blob_upload_string_or_bytes(
            connection_string=conn_str,
            container_name=container_name,
            blob_name=storage_file_basename,
            data=large_node,
        )
        storage_path = upload_instance["storage_path"]
        logger.debug(f"uploaded to: {storage_path}")
        return storage_path
    else:
        raise NotImplementedError(f"unimplemented storage option: '{storage_option}'")


def export_abox(onto: Ontology) -> Graph:
    """
    Create an RDFLib Graph of just the ABox (instances) from an Owlready2 ontology.

    For each individual the graph contains:

    * All asserted ``rdf:type`` triples for its OWL classes, or
      ``rdf:type owl:Thing`` if no explicit class is present.
    * All property assertions (data and object).

    Notes
    -----
    This is deliberately ABox-only and may be **lossy** with respect to
    the original TBox (no subclass axioms, domain/range, etc.).
    """
    abox = Graph()
    ns = Namespace(onto.base_iri)
    abox.bind("abstrax", ns)
    for ind in onto.individuals():
        subj = URIRef(ind.iri)
        if ind.is_a:
            for cls in ind.is_a:
                if hasattr(cls, "iri"):
                    abox.add((subj, RDF.type, URIRef(cls.iri)))
        else:
            abox.add((subj, RDF.type, URIRef(Thing.iri)))
        for prop in ind.get_properties():
            for obj in prop[ind]:
                obj_ref = URIRef(obj.iri) if hasattr(obj, "iri") else Literal(obj)
                abox.add((subj, URIRef(prop.iri), obj_ref))
    return abox


def stream_ntriples(onto: Ontology) -> Generator[str, None, None]:
    """
    Stream N-Triples lines from an ontology using an in-memory BytesIO.
    """
    buf = io.BytesIO()
    onto.save(file=buf, format="ntriples")
    buf.seek(0)
    for raw_line in buf:
        yield raw_line.decode("utf-8").rstrip("\n")


def parse_ntriples_line(line: str):
    """
    Parse a single N-Triples line into (subject, predicate, object).

    Raises :class:`rdflib.plugins.parsers.ntriples.ParseError` on
    malformed lines.
    """
    g = Graph()
    try:
        g.parse(data=line + "\n", format="nt")
    except ParseError:
        raise
    for s, p, o in g:
        return s, p, o
    raise ParseError(f"Could not extract triple from line: {line}")


def has_blank_node(lines: List[str]) -> bool:
    """
    Return True if any triple contains a blank node in subject or object.

    The push pipeline does not attempt to normalise or skolemise blank
    nodes; instead it fails fast when they are detected.
    """
    for ln in lines:
        try:
            s, _, o = parse_ntriples_line(ln)
            if isinstance(s, BNode) or isinstance(o, BNode):
                return True
        except ParseError:
            raise
    return False


def encode_string_to_uuid(input_string: str) -> str:
    """
    Convert an arbitrary string to a URL-safe, UUID-like string.

    The same input string will always produce the same output, and
    different input strings will produce different outputs.
    """
    hash_digest = hashlib.sha256(input_string.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=hash_digest[:16]))


def transform_triples(
    triple_dots: list[str],
    owlready_onto: Ontology,
    max_literal_size: Optional[int] = None,
) -> list[str]:
    """
    Offload large literals to Azure Blob Storage.

    For any triple whose object is a :class:`rdflib.Literal` and whose
    predicate is a known data property, if the UTF-8 encoded *lexical
    value* exceeds ``max_literal_size`` bytes, the literal is replaced
    with a short URL and the original value is stored externally.

    The record stored in Azure is:

    .. code-block:: json

        {
          "datatype": "<datatype IRI>",
          "value": "<plain lexical value>"
        }

    On pull, :func:`pydowl.sparql.pull.pull_owlready_from_sparql_endpoint`
    downloads the record, reconstructs the Python value using
    :mod:`pydowl.data_type` / builtins, and assigns it back to the
    corresponding data property.

    Notes
    -----
    If ``max_literal_size`` is ``None``, the module-level
    :data:`LARGE_LITERAL_THRESHOLD` is used. This allows tests to
    monkeypatch the threshold at runtime.
    """
    if max_literal_size is None:
        max_literal_size = LARGE_LITERAL_THRESHOLD

    data_props = {str(p.iri) for p in owlready_onto.data_properties()}
    out: list[str] = []
    for line in triple_dots:
        try:
            s, p, o = parse_ntriples_line(line)
        except ParseError:
            raise

        p_iri = str(p)
        if p_iri in data_props and isinstance(o, Literal):
            value = str(o)
            size = len(value.encode("utf-8"))
            if size > max_literal_size:
                dt_iri = (
                    str(o.datatype)
                    if o.datatype is not None
                    else "http://www.w3.org/2001/XMLSchema#string"
                )
                record = {"datatype": dt_iri, "value": value}
                record_json = json.dumps(record, separators=(",", ":"), sort_keys=True)
                blob_name = encode_string_to_uuid(record_json) + LARGE_NODE_IRI_SUFFIX
                payload = record_json.encode("utf-8")
                url = upload_large_node(payload, blob_name)
                new_line = (
                    f"{s.n3()} {p.n3()} "
                    f'"{url}"^^<http://www.w3.org/2001/XMLSchema#string> .'
                )
                out.append(new_line)
                continue

        out.append(line)
    return out


def batch_lines(
    stream: Generator[str, None, None], size: int
) -> Generator[List[str], None, None]:
    """
    Batch a line stream into lists of at most ``size`` lines.
    """
    buf: List[str] = []
    for ln in stream:
        buf.append(ln)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


class PushError(Exception):
    """Raised for irrecoverable push failures (e.g. blank nodes, max retries)."""


def push_owlready_to_sparql(
    onto: Ontology,
    endpoint: SparqlEndpoint,
    graph_iri: Optional[str] = None,
    abox_only: bool = False,
    chunk_size: int = 1000,
    retries: int = 3,
) -> None:
    """
    Push ontology (or ABox only) triples in chunks to a SPARQL endpoint.

    Parameters
    ----------
    onto:
        Owlready2 ontology to serialise.
    endpoint:
        :class:`SparqlEndpoint` to send updates to.
    graph_iri:
        Target named graph IRI. Defaults to ``onto.base_iri``.
    abox_only:
        If True, only instance (ABox) triples are sent.
    chunk_size:
        Maximum number of triples per SPARQL UPDATE batch.
    retries:
        Number of retry attempts per batch before failing with
        :class:`PushError`.
    """
    target = graph_iri or onto.base_iri

    # Prepare triple stream
    if abox_only:
        abox_graph = export_abox(onto)
        buf = io.BytesIO()
        abox_graph.serialize(destination=buf, format="nt", encoding="utf-8")
        buf.seek(0)

        def _abox_gen():
            for raw in buf:
                yield raw.decode("utf-8").rstrip("\n")

        stream = _abox_gen()
    else:
        stream = stream_ntriples(onto)

    functional_iris = {
        str(p.iri) for p in onto.properties() if FunctionalProperty in p.ancestors()
    }

    for batch in batch_lines(stream, chunk_size):
        if has_blank_node(batch):
            logger.error("Blank nodes detected; aborting push.")
            raise PushError("Blank nodes in data")
        batch = transform_triples(batch, onto)

        non_func: list[str] = []
        func_ins: list[str] = []
        func_del: list[str] = []

        for idx, triple in enumerate(batch):
            try:
                s, p, _ = parse_ntriples_line(triple)
            except ParseError:
                continue
            iri = str(p)
            if iri in functional_iris:
                func_ins.append(triple)
                func_del.append(f"{s.n3()} {p.n3()} ?old{idx} .")
            else:
                non_func.append(triple)

        if non_func:
            q = f"INSERT DATA {{ GRAPH <{target}> {{ {' '.join(non_func)} }} }}"
            for i in range(1, retries + 1):
                try:
                    endpoint.update(q)
                    break
                except Exception as e:  # pragma: no cover - depends on endpoint
                    logger.warning(f"Non-functional attempt {i} failed: {e}")
                    if i == retries:
                        raise PushError(
                            "Max retries for non-functional triples reached"
                        )

        if func_ins:
            ins = " ".join(func_ins)
            dele = " ".join(func_del)
            q = (
                f"DELETE {{ GRAPH <{target}> {{ {dele} }} }} "
                f"INSERT {{ GRAPH <{target}> {{ {ins} }} }} "
                f"WHERE {{ GRAPH <{target}> {{ OPTIONAL {{ {dele} }} }} }}"
            )
            for i in range(1, retries + 1):
                try:
                    endpoint.update(q)
                    break
                except Exception as e:  # pragma: no cover - depends on endpoint
                    logger.warning(f"Functional upsert attempt {i} failed: {e}")
                    if i == retries:
                        raise PushError("Max retries for functional triples reached")


# Backward compatibility alias
push_owlready_to_sparql_endpoint = push_owlready_to_sparql


def push_pyd_to_sparql_endpoint(
    pyd_instance: PydOwlClass,
    endpoint: SparqlEndpoint,
    abox_only: bool = False,
    tbox_ontology_path: Optional[str] = None,
) -> None:
    """
    Convenience: push a :class:`PydOwlClass` instance to a SPARQL endpoint.

    If ``tbox_ontology_path`` is provided, it should be the path/IRI of
    an existing ontology that defines the required classes/properties.
    In that case, dynamic TBox creation is disabled and a mismatch will
    surface as a :class:`pydowl.base_class.KgPushError`.

    If ``tbox_ontology_path`` is omitted, a temporary ontology with a
    flat TBox is created dynamically (``dynamic_tbox=True``).
    """
    world = World()
    if tbox_ontology_path:
        onto = world.get_ontology(tbox_ontology_path).load()
        iri = onto.base_iri
        dynamic = False
    else:
        iri = PYD_PUSH_TO_SPARQL_TEMPORARY_ONTOLOGY_IRI
        onto = world.get_ontology(iri)
        dynamic = True

    pyd_instance.push_owlready(onto, dynamic_tbox=dynamic)
    push_owlready_to_sparql(
        onto, endpoint, graph_iri=onto.base_iri, abox_only=abox_only
    )

    # Clean up the temporary ontology from the World to avoid leaks.
    for ent in list(onto.individuals()):
        destroy_entity(ent)
    for cls in list(onto.classes()):
        destroy_entity(cls)
    for prop in list(onto.properties()):
        destroy_entity(prop)
    try:
        key = iri[:-1] if iri.endswith("#") else iri
        world._namespaces.pop(key, None)
    except Exception as ex:  # pragma: no cover - defensive
        logger.warning(f"Namespace cleanup failed: {ex}")
