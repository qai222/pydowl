import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as _np
import owlready2
import pydantic_numpy.typing as pnd
import pytest
from rdflib import URIRef, Literal

from pydowl import PydOwlClass
from pydowl.sparql import pull_pyd_from_sparql_endpoint
from pydowl.sparql.push import (
    push_owlready_to_sparql_endpoint,
    transform_triples,
    parse_ntriples_line,
)

pytest_plugins = ["tests.fixtures.sparql_fixtures"]


# ──────────────────────────────────────────────────────────────────────────
# Module‐level PydOwlClass subclasses for importable discriminator resolution
# ──────────────────────────────────────────────────────────────────────────


class JsonLargeJ(PydOwlClass):
    meta: dict[str, Any]


class NumpyA(PydOwlClass):
    arr: Optional[pnd.Np2DArray]


class DateTimeE(PydOwlClass):
    ts: Optional[datetime] = None


class LNChild(PydOwlClass):
    data: dict[str, Any]


class LNParent(PydOwlClass):
    child: Optional[LNChild]
    extra: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────
# Azure‐mode fixture: "fake" vs "real"
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["fake", "real"], autouse=True)
def azure_mode(request, monkeypatch):
    """
    Parametrized fixture: for "fake" we stub out Azure upload/download
    to an in-memory dict; for "real" we rely on actual Azure creds. If
    real creds are missing, skip that half.
    """
    mode = request.param

    if mode == "fake":
        store: Dict[str, bytes] = {}

        def fake_upload_string_or_bytes(
            *, connection_string, container_name, blob_name, data
        ):
            payload = data if isinstance(data, bytes) else data.encode("utf-8")
            store[blob_name] = payload
            return {
                "blob_name": blob_name,
                "container_name": container_name,
                "storage_path": f"https://fake.blob/{blob_name}",
            }

        def fake_download_bytes_or_str(blob_url: str, sas_token: str):
            name = blob_url.rsplit("/", 1)[-1]
            return store[name]

        # Force off-load threshold to 0 so every literal is off-loaded
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "fake-conn")
        monkeypatch.setenv("AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN", "fake-sas")
        monkeypatch.setattr(
            "pydowl.sparql.push.LARGE_LITERAL_THRESHOLD", 0, raising=False
        )
        monkeypatch.setattr(
            "pydowl.sparql.push.blob_upload_string_or_bytes",
            fake_upload_string_or_bytes,
            raising=True,
        )
        monkeypatch.setattr(
            "pydowl.sparql.pull.blob_download_bytes_or_str",
            fake_download_bytes_or_str,
            raising=True,
        )

    else:  # real Azure
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        sas = os.getenv("AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN")
        if not conn or not sas:
            pytest.skip("Real Azure credentials not set; skipping real-Azure tests")
        # still override threshold=0 to guarantee offload in tests
        monkeypatch.setattr(
            "pydowl.sparql.push.LARGE_LITERAL_THRESHOLD", 0, raising=False
        )

    yield


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("sparql_endpoint")
def test_json_large_literal_roundtrip_simple(test_onto, sparql_endpoint):
    """
    A large JSON literal (> threshold) off-loads to Azure and pulls back intact.
    """
    big = {"text": "A" * 100_000, "nums": list(range(10_000))}

    j1 = JsonLargeJ(identifier="j1", meta=big)
    j1.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    j2 = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "j1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
    )
    assert isinstance(j2, JsonLargeJ)
    assert j2.meta == big


@pytest.mark.usefixtures("sparql_endpoint")
def test_numpy_array_large_literal_roundtrip(test_onto, sparql_endpoint):
    """
    A large NumPy array off-loads and pulls back as ndarray.
    """
    arr = _np.arange(10_000, dtype=_np.int32).reshape((100, 100))
    a1 = NumpyA(identifier="a1", arr=arr)
    a1.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    a2 = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "a1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
    )
    assert isinstance(a2, NumpyA)
    assert isinstance(a2.arr, _np.ndarray)
    assert _np.array_equal(a2.arr, arr)


@pytest.mark.usefixtures("sparql_endpoint")
def test_datetime_large_literal_roundtrip(test_onto, sparql_endpoint):
    """
    A large datetime literal off-loads and pulls back as a datetime.
    """
    now = datetime.now(timezone.utc).replace(microsecond=123456)
    e1 = DateTimeE(identifier="e1", ts=now)
    e1.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    e2 = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "e1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
    )
    assert isinstance(e2, DateTimeE)
    assert isinstance(e2.ts, datetime)
    assert e2.ts.isoformat() == now.isoformat()


@pytest.mark.usefixtures("sparql_endpoint")
def test_nested_model_with_large_nodes(test_onto, sparql_endpoint):
    """
    Nested PydOwlClass with two large fields—both off-load and pull back correctly.
    """
    # Register OWL classes
    with test_onto:
        type("LNChild", (owlready2.Thing,), {})
        type("LNParent", (owlready2.Thing,), {})

    blob1 = {"text": "X" * 50_000, "vals": list(range(5_000))}
    blob2 = {"foo": ["Y"] * 20_000}

    c1 = LNChild(identifier="c1", data=blob1)
    p1 = LNParent(identifier="p1", child=c1, extra=blob2)
    p1.push_owlready(test_onto, dynamic_tbox=True)
    push_owlready_to_sparql_endpoint(test_onto, sparql_endpoint, abox_only=False)

    p2 = pull_pyd_from_sparql_endpoint(
        ind_iri=test_onto.base_iri + "p1",
        graph_iri=test_onto.base_iri,
        sparql_ep=sparql_endpoint,
    )
    assert isinstance(p2, LNParent)
    assert isinstance(p2.child, LNChild)
    assert p2.child.data == blob1
    assert p2.extra == blob2


def test_transform_triples_record_contents(test_onto):
    """
    transform_triples must replace a JSON triple with a blob‐URL literal.
    """
    # 1) Register has_meta so transform sees a data property
    with test_onto:

        class has_meta(owlready2.DataProperty, owlready2.FunctionalProperty):
            pass

    subj = URIRef("http://example.org/s")
    pred = URIRef(f"{test_onto.base_iri}has_meta")
    lit = Literal(
        {"a": 1, "b": 2},
        datatype=URIRef("http://pydowl.org/dtype#JsonString"),
    )
    triple = f"{subj.n3()} {pred.n3()} {lit.n3()} ."

    # force off-load
    out = transform_triples([triple], test_onto, max_literal_size=0)
    assert len(out) == 1

    s2, p2, o2 = parse_ntriples_line(out[0])
    assert str(s2) == str(subj)
    assert str(p2) == str(pred)
    url = o2.toPython()
    assert isinstance(url, str)
    assert url.startswith("https://fake.blob/") or url.startswith("https://"), (
        f"got {url!r}"
    )
