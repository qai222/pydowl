"""
pydowl.sparql.pull
==================

Pull `PydOwlClass` instances from a SPARQL endpoint.

Design overview
---------------

This module provides **two tiers** of pulling data from a SPARQL 1.1
endpoint into :class:`pydowl.base_class.PydOwlClass` instances.

Tier 1 – full-graph / ontology-centric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`pull_owlready_from_sparql_endpoint` loads an entire named graph
  into an Owlready2 :class:`~owlready2.Ontology`, resolving any
  large-literal placeholders from Azure Blob storage.
* :func:`pull_pyd_from_sparql_full` then locates a specific individual
  by IRI and delegates to
  :meth:`pydowl.base_class.PydOwlClass.pull_owlready`, which uses the
  *local* TBox (classes and property declarations) to reconstruct the
  Pydantic object graph.

Use when:

* You want a full ontology for reasoning, debugging, or export.
* The named graph is not so large that loading it in memory is
  problematic.
* You want maximum fidelity to the stored OWL/TBox.

Tier 2 – ABox-only / schema-centric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`pull_pyd_from_sparql_abox` treats the SPARQL endpoint as a
  **fact store** and uses the **Pydantic schema (pydowl TBox)** as the
  *only* ground truth about classes and properties.
* It issues small SPARQL SELECT queries per field to fetch exactly the
  triples needed for a given subject (and recursively its neighbours),
  then maps them directly into a Pydantic `data` dict according to
  :class:`pydowl.field_type.FieldTypeCategory` and the
  ``has_<field>`` naming convention.

Use when:

* You want to hydrate a Pydantic object graph for one or a few
  individuals as part of an API or application flow.
* The named graph is large and you do **not** want to load it all at
  once.
* You trust the Pydantic schema as the canonical TBox and treat the
  SPARQL endpoint primarily as a data store.

Assumptions
-----------

* The **only ground-truth TBox** is defined in pydowl (Pydantic models).
  The SPARQL endpoint’s TBox is treated as best-effort but not
  authoritative.
* Named graph IRI is equal to the ontology base IRI used by
  :class:`pydowl.base_class.PydOwlClass`, and properties follow the
  ``has_<attribute>`` naming convention (e.g. ``has_name`` for
  ``name``).
* Individuals are IRI-ed as ``<base_iri><identifier>``; identifiers are
  recovered by stripping ``base_iri`` from the subject IRI whenever
  possible.

Limitations
-----------

* Tier 1 (`"full"` mode) always loads the entire named graph into
  Owlready; this may not be suitable for very large graphs.
* Tier 2 (`"abox"` mode) currently follows relationships only through
  fields declared as :class:`PydOwlClass` (or lists thereof); additional
  arbitrary edges in the graph will be ignored unless modelled in the
  Pydantic schema.
"""

import io
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Set, cast

from SPARQLWrapper import TURTLE
from loguru import logger
from owlready2 import (
    World,
    Ontology,
    DataProperty,
    ObjectProperty,
    FunctionalProperty,
)
from rdflib import Graph
from rdflib.namespace import RDF

from pydowl.base_class import PydOwlClass, KgPullError
from pydowl.base_class import _resolve_class_from_tag
from pydowl.data_type import (
    OWLREADY2_BUILTIN_DATATYPE_TABLE,
    CUSTOM_OWL_DATATYPE_TABLE,
    get_datatype_parser,
)
from pydowl.field_type import (
    FieldTypeCategory,
    identify_field_type_category,
    get_inner_type,
)
from pydowl.sparql.endpoint import SparqlEndpoint
from pydowl.sparql.utils_azure import blob_download_bytes_or_str
from .settings import (
    ENV_AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN,
    LARGE_NODE_IRI_SUFFIX,
)

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

# Azure SAS token (if set); if missing, large-node downloads will fail
# loudly when encountered.
AZURE_SAS = os.getenv(ENV_AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN, "")

# ──────────────────────────────────────────────────────────────────────────
# Cache for TBox introspection (used by examine_properties)
# ──────────────────────────────────────────────────────────────────────────

_examine_cache: Dict[
    str, Tuple[List[str], List[str], List[str], Dict[str, List[Any]]]
] = {}


def clear_examine_cache() -> None:
    """
    Clear the internal TBox-introspection cache.

    Call this after you’ve programmatically modified an Owlready2
    ontology’s class/property definitions to ensure
    :func:`examine_properties` will re-scan.
    """
    _examine_cache.clear()


def examine_properties(
    onto: Ontology,
) -> Tuple[List[str], List[str], List[str], Dict[str, List[Any]]]:
    """
    Introspect the OWL TBox (once per ontology base IRI) to classify:

    * data_props: IRIs of :class:`owlready2.DataProperty` classes
    * func_obj_props: IRIs of functional :class:`owlready2.ObjectProperty`
    * nonfunc_obj_props: IRIs of non-functional ObjectProperty classes
    * dt_ranges: mapping DataProperty IRI → its declared range list
    """
    cache_id = onto.base_iri
    if cache_id in _examine_cache:
        return _examine_cache[cache_id]

    data_props: List[str] = []
    func_obj: List[str] = []
    nonfunc_obj: List[str] = []
    dt_ranges: Dict[str, List[Any]] = {}

    for prop in onto.properties():
        if issubclass(prop, DataProperty):
            data_props.append(prop.iri)
            dt_ranges[prop.iri] = list(prop.range)
        elif issubclass(prop, ObjectProperty):
            if issubclass(prop, FunctionalProperty):
                func_obj.append(prop.iri)
            else:
                nonfunc_obj.append(prop.iri)

    _examine_cache[cache_id] = (data_props, func_obj, nonfunc_obj, dt_ranges)
    return _examine_cache[cache_id]


# ──────────────────────────────────────────────────────────────────────────
# Helper: SPARQL CONSTRUCT → Owlready2 load + large-literal resolve
# ──────────────────────────────────────────────────────────────────────────


def pull_owlready_from_sparql_endpoint(
    graph_iri: str, sparql_ep: SparqlEndpoint, construct_q: Optional[str] = None
) -> Ontology:
    """
    Load a named graph from a SPARQL endpoint into an Owlready2 ontology.

    Steps
    -----
    1. Issue a CONSTRUCT query (default: all triples in the named graph)
       and fetch Turtle.
    2. Parse into an :class:`rdflib.Graph` and serialise to N-Triples
       bytes.
    3. Load those bytes into a fresh Owlready2 :class:`World` /
       :class:`Ontology`.
    4. Ensure a ``has_pydowl_type`` functional data property exists
       (purely as a schema convenience; values are not invented).
    5. Resolve any large-literal placeholders that were previously
       offloaded to Azure Blob Storage by
       :func:`pydowl.sparql.push.transform_triples`.
    """
    if not construct_q:
        construct_q = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE  {{ GRAPH <{graph_iri}> {{ ?s ?p ?o }} }}
        """

    # Fetch Turtle via SPARQLWrapper directly (we want CONSTRUCT as text)
    wrapper = sparql_ep.setup_wrapper(method="POST", return_format=TURTLE)
    wrapper.setQuery(construct_q)
    turtle_text = cast(str, wrapper.query().convert())

    # rdflib parse → serialize bytes
    g = Graph()
    g.parse(data=turtle_text, format="turtle")
    nt_bytes = g.serialize(format="nt").encode("utf-8")
    buf = io.BytesIO(nt_bytes)

    # Owlready2 load from bytes
    world = World()
    onto = world.get_ontology(graph_iri)
    onto.load(fileobj=buf, format="ntriples")

    # Ensure has_pydowl_type exists as a functional data property
    tag_iri = onto.base_iri + "has_pydowl_type"
    if not onto.search_one(iri=tag_iri):
        logger.debug("no `has_pydowl_type` found, declaring it")
        with onto:

            class has_pydowl_type(DataProperty, FunctionalProperty):
                range = [str]

    # Resolve large-node placeholders
    for dp in onto.data_properties():
        for subj, obj in dp.get_relations():
            if isinstance(obj, str) and obj.endswith(LARGE_NODE_IRI_SUFFIX):
                logger.debug(f"Resolving large node placeholder: {obj}")
                blob_url = obj
                raw = blob_download_bytes_or_str(blob_url, sas_token=AZURE_SAS)
                record = json.loads(raw)
                dt_iri = record["datatype"]
                value = record["value"]

                if dt_iri in OWLREADY2_BUILTIN_DATATYPE_TABLE:
                    # For built-in XSD types, store the plain Python scalar
                    conv = OWLREADY2_BUILTIN_DATATYPE_TABLE[dt_iri]
                    try:
                        py_val = conv(value)
                    except Exception:
                        py_val = value
                elif dt_iri in CUSTOM_OWL_DATATYPE_TABLE:
                    # For pydowl custom datatypes, store the *wrapper* object
                    # (e.g. JsonStringDatatype, NpArray2dDatatype, DateTimeDatatype).
                    # PydOwlClass.pull_owlready will unwrap .val to the Python object.
                    dt_cls = CUSTOM_OWL_DATATYPE_TABLE[dt_iri]
                    py_val = get_datatype_parser(dt_cls)(value)
                else:
                    # Unknown datatype – fall back to raw string
                    py_val = value

                setattr(subj, dp.name, py_val)

    return onto


# ──────────────────────────────────────────────────────────────────────────
# Optional SELECT-based helpers (advanced)
# ──────────────────────────────────────────────────────────────────────────


def get_individual_classes(
    ind_iri: str, graph_iri: str, sparql_ep: SparqlEndpoint, leaf_only: bool = False
) -> List[str]:
    """
    SELECT rdf:type IRIs for <ind_iri> in <graph_iri>.

    If ``leaf_only=True``, only return types that have no subclasses
    asserted in the same graph.
    """
    q = f"""
    PREFIX rdf: <{RDF}>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?class (IF(BOUND(?sub), "Super","Leaf") AS ?kind)
    WHERE {{
      GRAPH <{graph_iri}> {{
        <{ind_iri}> rdf:type ?class .
        FILTER(?class != owl:NamedIndividual)
        OPTIONAL {{ ?sub rdfs:subClassOf ?class }}
      }}
    }}
    """
    rows = sparql_ep.query_select(q)
    out: List[str] = []
    for r in rows:
        if not leaf_only or r["kind"]["value"] == "Leaf":
            out.append(r["class"]["value"])
    return out


def get_individual_properties(
    ind_iri: str, graph_iri: str, sparql_ep: SparqlEndpoint
) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
    """
    Retrieve all property‐value pairs for <ind_iri>.

    Returns a mapping ``propIRI → list of (lexicalValue, propType, datatypeIRI)``,
    where ``propType`` is ``"object"`` or ``"data"``.
    """
    q = f"""
    PREFIX rdf: <{RDF}>

    SELECT ?p ?o (datatype(?o) AS ?dt)
           (IF(
              EXISTS {{ ?p a <http://www.w3.org/2002/07/owl#ObjectProperty> }},
              "object","data"
            ) AS ?pt)
    WHERE {{
      GRAPH <{graph_iri}> {{ <{ind_iri}> ?p ?o . }}
    }}
    """
    rows = sparql_ep.query_select(q)
    out: Dict[str, List[Tuple[str, str, Optional[str]]]] = defaultdict(list)
    for r in rows:
        prop_iri = r["p"]["value"]
        lex = r["o"]["value"]
        dt = r.get("dt", {}).get("value")
        pt = r["pt"]["value"]
        out[prop_iri].append((lex, pt, dt))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Schema-based CONSTRUCT pattern generator (currently unused)
# ──────────────────────────────────────────────────────────────────────────


def generate_construct_pattern(
    cls: Type[PydOwlClass],
    graph_iri: str,
    var: str = "?s",
    seen: Optional[Set[Type[PydOwlClass]]] = None,
    depth: int = 2,
) -> str:
    """
    Recursively generate a SPARQL CONSTRUCT/WHERE pattern for all fields
    of ``cls`` up to ``depth``.

    The pattern assumes the ``has_<field>`` property naming convention
    and uses ``var`` as the subject variable for ``cls`` instances.

    Note
    ----
    This helper is currently not used by the main pull functions but is
    kept for potential future “schema-based subset” CONSTRUCT queries.
    """
    if seen is None:
        seen = set()
    if cls in seen or depth < 0:
        return ""
    seen.add(cls)
    lines: List[str] = []
    for name, field in cls.model_fields.items():
        prop_iri = f"<{graph_iri}has_{name}>"
        obj_var = f"?{name}"
        lines.append(f"    {var} {prop_iri} {obj_var} .")
        cat = identify_field_type_category(field, name)
        if cat in (FieldTypeCategory.OPTIONAL_PYD_CLS, FieldTypeCategory.LIST_PYD_CLS):
            inner = get_inner_type(field)
            if issubclass(inner, PydOwlClass):
                nested = generate_construct_pattern(
                    cast(Type[PydOwlClass], inner),
                    graph_iri,
                    var=obj_var,
                    seen=seen,
                    depth=depth - 1,
                )
                if nested:
                    lines.append(nested)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Tier 1: full-graph PydOwlClass pull
# ──────────────────────────────────────────────────────────────────────────


def pull_pyd_from_sparql_full(
    ind_iri: str,
    graph_iri: str,
    sparql_ep: SparqlEndpoint,
    pyd_cls: Optional[Type[PydOwlClass]] = None,
) -> PydOwlClass:
    """
    Full-graph, ontology-centric pull.

    Loads the entire named graph into an Owlready2 ontology via
    :func:`pull_owlready_from_sparql_endpoint` and then delegates to
    :meth:`PydOwlClass.pull_owlready`.

    Use this when you want:

    * a complete ontology for reasoning or debugging, or
    * the graph is small enough that loading it is acceptable.

    Assumptions
    -----------
    * The named graph is small/medium enough to load in its entirety.
    * The graph's TBox is at least broadly consistent with the Pydantic
      schema (class and property IRIs match what pydowl expects).
    """
    # Infer PydOwlClass if not provided
    if pyd_cls is None:
        sel_q = f"""
        SELECT ?tag WHERE {{
          GRAPH <{graph_iri}> {{
            <{ind_iri}> <{graph_iri}has_pydowl_type> ?tag .
          }}
        }}
        """
        rows = sparql_ep.query_select(sel_q)
        if not rows:
            raise KgPullError(
                f"No `has_pydowl_type` triple found for individual {ind_iri}"
            )
        tag_raw = rows[0]["tag"]["value"]
        # Decode large-literal placeholders, if any
        tag_val, dt_iri = _maybe_resolve_large_literal(tag_raw)
        tag = tag_val  # discriminator is always a string
        rc = _resolve_class_from_tag(tag)
        if not rc or not issubclass(rc, PydOwlClass):
            raise KgPullError(f"Cannot resolve class for discriminator `{tag}`")
        pyd_cls = cast(Type[PydOwlClass], rc)

    # Load entire named graph
    onto = pull_owlready_from_sparql_endpoint(graph_iri, sparql_ep)
    owl_ind = onto.search_one(iri=ind_iri)
    if not owl_ind:
        raise KgPullError(
            f"Individual {ind_iri} not found in ontology loaded from {graph_iri}"
        )
    return pyd_cls.pull_owlready(onto, owl_ind)


# ──────────────────────────────────────────────────────────────────────────
# Tier 2: ABox-only, schema-centric PydOwlClass pull
# ──────────────────────────────────────────────────────────────────────────


def _decode_scalar_from_sparql(value: str, dt_iri: Optional[str]) -> Any:
    """
    Decode a scalar literal from SPARQL JSON into a Python value.

    Uses:

    * :data:`OWLREADY2_BUILTIN_DATATYPE_TABLE` for known XSD types.
    * :data:`CUSTOM_OWL_DATATYPE_TABLE` + :func:`get_datatype_parser`
      for pydowl custom datatypes.
    * Falls back to the raw string if no mapping is known.
    """
    if dt_iri and dt_iri in OWLREADY2_BUILTIN_DATATYPE_TABLE:
        conv = OWLREADY2_BUILTIN_DATATYPE_TABLE[dt_iri]
        try:
            return conv(value)
        except Exception:
            return value
    if dt_iri and dt_iri in CUSTOM_OWL_DATATYPE_TABLE:
        dt_cls = CUSTOM_OWL_DATATYPE_TABLE[dt_iri]
        wrapped = get_datatype_parser(dt_cls)(value)
        return getattr(wrapped, "val", wrapped)
    return value


def _maybe_resolve_large_literal(value: str) -> Tuple[str, Optional[str]]:
    """
    If ``value`` looks like a large-node placeholder (endswith
    LARGE_NODE_IRI_SUFFIX), fetch the corresponding record from Azure
    and return (actual_value, datatype_iri). Otherwise, return
    (value, None).
    """
    if isinstance(value, str) and value.endswith(LARGE_NODE_IRI_SUFFIX):
        raw = blob_download_bytes_or_str(value, sas_token=AZURE_SAS)
        record = json.loads(raw)
        return record["value"], record["datatype"]
    return value, None


def _iri_to_identifier(ind_iri: str, graph_iri: str) -> str:
    """
    Derive a pydowl identifier from an individual IRI.

    Prefers stripping ``graph_iri`` prefix when present, otherwise falls
    back to the last fragment after ``#`` or ``/``.
    """
    if ind_iri.startswith(graph_iri):
        return ind_iri[len(graph_iri) :]
    frag = ind_iri.rsplit("#", 1)[-1]
    frag = frag.rsplit("/", 1)[-1]
    return frag


def pull_pyd_from_sparql_abox(
    ind_iri: str,
    graph_iri: str,
    sparql_ep: SparqlEndpoint,
    pyd_cls: Optional[Type[PydOwlClass]] = None,
    *,
    max_depth: int = 2,
) -> PydOwlClass:
    """
    ABox-only, schema-centric pull.

    This function treats the SPARQL endpoint as a **fact store** and
    uses the Pydantic schema (pydowl TBox) as the *only* ground truth
    about classes and properties.

    Behaviour
    ---------
    * For a given ``pyd_cls`` and subject IRI, it inspects the Pydantic
      fields and their :class:`FieldTypeCategory` values.
    * For each field it knows the corresponding property IRI
      ``<graph_iri>has_<field_name>``.
    * It then issues small SPARQL SELECT queries per field to fetch
      exactly the triples needed for that field, decodes the SPARQL
      values into Python, and builds a ``data`` dict.
    * Nested :class:`PydOwlClass` fields (and lists thereof) are
      followed recursively up to ``max_depth``, with cycle-safe reuse of
      already-visited nodes.
    * Finally, it calls :meth:`PydOwlClass.from_data` to construct or
      update instances, so the global :class:`PydOwlRegistry` is
      respected.

    Use when:

    * You want to hydrate a Pydantic object graph for one or a few
      individuals, as part of an application or API layer.
    * The named graph is large and you want to avoid loading the entire
      graph into memory.
    * You assume that the Pydantic model accurately reflects the
      intended TBox, and the SPARQL endpoint contains the corresponding
      ABox triples.
    """
    # 1) Infer class from `has_pydowl_type` if not provided
    if pyd_cls is None:
        sel_q = f"""
        SELECT ?tag WHERE {{
          GRAPH <{graph_iri}> {{
            <{ind_iri}> <{graph_iri}has_pydowl_type> ?tag .
          }}
        }}
        """
        rows = sparql_ep.query_select(sel_q)
        if not rows:
            raise KgPullError(
                f"No `has_pydowl_type` triple found for individual {ind_iri}"
            )
        tag_raw = rows[0]["tag"]["value"]
        # Decode large-literal placeholders, if any
        tag_val, dt_iri = _maybe_resolve_large_literal(tag_raw)
        tag = tag_val
        rc = _resolve_class_from_tag(tag)
        if not rc or not issubclass(rc, PydOwlClass):
            raise KgPullError(f"Cannot resolve class for discriminator `{tag}`")
        pyd_cls = cast(Type[PydOwlClass], rc)

    visited: Dict[Tuple[Type[PydOwlClass], str], PydOwlClass] = {}

    def walk(cls: Type[PydOwlClass], subj_iri: str, depth: int) -> PydOwlClass:
        ident = _iri_to_identifier(subj_iri, graph_iri)
        key = (cls, ident)
        if key in visited:
            return visited[key]

        # Create (or reuse) instance via from_data and register early to
        # break cycles.
        instance = cls.from_data({"identifier": ident})
        visited[key] = instance

        if depth > max_depth:
            return instance

        update_data: Dict[str, Any] = {}

        for field_name, field_info in cls.model_fields.items():
            if field_name in ("identifier", "pydowl_version"):
                continue

            category = identify_field_type_category(field_info, field_name)
            prop_iri = f"{graph_iri}has_{field_name}"

            # Scalar-ish fields
            if category in (
                FieldTypeCategory.OPTIONAL_PY_LITERAL,
                FieldTypeCategory.OPTIONAL_DATETIME,
                FieldTypeCategory.OPTIONAL_STR_ENUM,
                FieldTypeCategory.OPTIONAL_INT_ENUM,
                FieldTypeCategory.PY_JSON,
                FieldTypeCategory.OPTIONAL_NPND_ARRAY,
                FieldTypeCategory.OPTIONAL_NP2D_ARRAY,
            ):
                q = f"""
                SELECT ?v WHERE {{
                  GRAPH <{graph_iri}> {{
                    <{subj_iri}> <{prop_iri}> ?v .
                  }}
                }} LIMIT 1
                """
                rows = sparql_ep.query_select(q)
                if not rows:
                    continue
                v_binding = rows[0]["v"]
                raw_val = v_binding["value"]
                dt_iri = v_binding.get("datatype")

                # Handle large-literal placeholders
                raw_val, dt_from_blob = _maybe_resolve_large_literal(raw_val)
                dt_eff = dt_from_blob or dt_iri

                py_val = _decode_scalar_from_sparql(raw_val, dt_eff)
                update_data[field_name] = py_val

            # Optional nested PydOwlClass
            elif category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                q = f"""
                SELECT ?child WHERE {{
                  GRAPH <{graph_iri}> {{
                    <{subj_iri}> <{prop_iri}> ?child .
                  }}
                }} LIMIT 1
                """
                rows = sparql_ep.query_select(q)
                if not rows:
                    continue
                child_iri = rows[0]["child"]["value"]
                inner_type = get_inner_type(field_info)
                if not (
                    isinstance(inner_type, type) and issubclass(inner_type, PydOwlClass)
                ):
                    continue
                child_inst = walk(
                    cast(Type[PydOwlClass], inner_type), child_iri, depth + 1
                )
                update_data[field_name] = child_inst

            # List of nested PydOwlClass
            elif category == FieldTypeCategory.LIST_PYD_CLS:
                q = f"""
                SELECT ?child WHERE {{
                  GRAPH <{graph_iri}> {{
                    <{subj_iri}> <{prop_iri}> ?child .
                  }}
                }}
                """
                rows = sparql_ep.query_select(q)
                if not rows:
                    continue
                inner_type = get_inner_type(field_info)
                if not (
                    isinstance(inner_type, type) and issubclass(inner_type, PydOwlClass)
                ):
                    continue
                items: List[PydOwlClass] = []
                for r in rows:
                    child_iri = r["child"]["value"]
                    child_inst = walk(
                        cast(Type[PydOwlClass], inner_type), child_iri, depth + 1
                    )
                    items.append(child_inst)
                update_data[field_name] = items

            else:
                # Reserved or unsupported categories are ignored here.
                continue

        if update_data:
            instance.update(**update_data)
        return instance

    return walk(pyd_cls, ind_iri, depth=0)


# ──────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────


def pull_pyd_from_sparql_endpoint(
    ind_iri: str,
    graph_iri: str,
    sparql_ep: SparqlEndpoint,
    pyd_cls: Optional[Type[PydOwlClass]] = None,
    *,
    mode: str = "full",
    max_depth: int = 2,
) -> PydOwlClass:
    """
    Unified entry point for pulling a :class:`PydOwlClass` from SPARQL.

    Parameters
    ----------
    ind_iri:
        IRI of the individual to pull.
    graph_iri:
        Named graph IRI (assumed equal to the ontology ``base_iri``).
    sparql_ep:
        SPARQL endpoint wrapper.
    pyd_cls:
        Optional PydOwlClass subclass override. If omitted, the
        discriminator ``has_pydowl_type`` is used to infer the class.
    mode:
        Either ``"full"`` or ``"abox"``:

        * ``"full"`` – Tier 1: load the entire named graph into
          Owlready2 and delegate to
          :func:`pull_pyd_from_sparql_full`. Use when you want a full
          ontology for reasoning / debugging.
        * ``"abox"`` – Tier 2: use schema-centric SPARQL SELECT queries
          to fetch just the ABox needed for this instance (and its
          neighbours up to ``max_depth``), then delegate to
          :func:`pull_pyd_from_sparql_abox`. Use in application code
          where you want a light-weight data pull.

    max_depth:
        Maximum recursion depth for nested PydOwlClass fields in
        ``"abox"`` mode. Ignored in ``"full"`` mode.
    """
    if mode == "abox":
        return pull_pyd_from_sparql_abox(
            ind_iri, graph_iri, sparql_ep, pyd_cls=pyd_cls, max_depth=max_depth
        )
    elif mode == "full":
        return pull_pyd_from_sparql_full(ind_iri, graph_iri, sparql_ep, pyd_cls=pyd_cls)
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'full' or 'abox')")
