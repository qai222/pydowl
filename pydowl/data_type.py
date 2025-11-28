import base64
import datetime
import io
import json
from typing import Callable, Final

import numpy as np
import pandas as pd
from owlready2 import declare_datatype, DatatypeClass

"""
pydowl.data_type
================
Custom OWL datatypes that bridge complex Python objects (JSON blobs,
NumPy arrays, datetimes) to plain RDF literals.

---------------------------------------------------------------------------
Datatype-mapping rationale
---------------------------------------------------------------------------
The mapping from **Pydantic field → OWL literal** is *surjective*: several
distinct Python objects (e.g. any valid JSON value) are collapsed onto a
single OWL literal whose datatype IRI is declared here. A
Knowledge-Graph (KG) stored as RDF/OWL is, by design, schemaless—triples
carry no intrinsic unit or array shape metadata beyond the datatype IRI
on the literal itself.

• **Push phase** – When serialising a Pydantic instance to OWL we
  *must* attach that datatype IRI so downstream reasoners and SPARQL
  clients can interpret the payload correctly. Owlready2 exposes this
  via :func:`declare_datatype`
  (<https://owlready2.readthedocs.io/en/latest/datatype.html>).

• **Pull phase** – When materialising OWL individuals back to Python we
  already know the target field types from the Pydantic model, so no
  extra hint is required; we simply look up the parser function in
  :data:`_PARSERS` and reconstruct the object.

Namespace choice
----------------
Custom datatypes are minted in the dedicated pydowl namespace
``http://pydowl.org/dtype#`` rather than under the XML Schema namespace
– this avoids clashing with standard XSD datatypes and keeps the
extension clearly separated.
"""

_PYDOWL_DTYPE_NS: Final[str] = "http://pydowl.org/dtype#"


class DatatypeParserError(LookupError):
    """Raised when no parser is registered for a given OWL datatype."""


def _bytes_to_b64(b: bytes) -> str:
    """Encode *bytes* to URL-safe Base64 without copying the buffer."""
    return base64.urlsafe_b64encode(memoryview(b)).decode()


def _b64_to_bytes(s: str) -> bytes:
    return base64.urlsafe_b64decode(s)


def _ndarray_to_bytes(arr: np.ndarray) -> bytes:
    """Compress an n-D array with ``np.savez_compressed``."""
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr)
    return buf.getvalue()


def _bytes_to_ndarray(b: bytes) -> np.ndarray:
    with np.load(io.BytesIO(b), allow_pickle=False) as data:
        return data["arr"].copy()


# ──────────────────────────────────────────────────────────────────────────
# Custom Datatype: JSON string
# ──────────────────────────────────────────────────────────────────────────


class JsonStringDatatype:
    """
    Wraps an arbitrary JSON-serialisable Python object.

    In OWL this is represented as an RDF literal with datatype IRI
    ``http://pydowl.org/dtype#JsonString``.
    """

    __slots__ = ("val",)
    iri: Final[str] = f"{_PYDOWL_DTYPE_NS}JsonString"

    def __init__(self, val):
        self.val = val


def _json_string_datatype_parser(s: str):
    return JsonStringDatatype(val=json.loads(s))


def _json_string_datatype_unparser(j: JsonStringDatatype):
    return json.dumps(j.val)


declare_datatype(
    JsonStringDatatype,
    JsonStringDatatype.iri,
    _json_string_datatype_parser,
    _json_string_datatype_unparser,
)


# ──────────────────────────────────────────────────────────────────────────
# Custom Datatype: n-D NumPy array
# ──────────────────────────────────────────────────────────────────────────


class NpArrayNdDatatype:
    """NumPy array of arbitrary dimensionality *except* 2-D (see below)."""

    __slots__ = ("val",)
    iri: Final[str] = f"{_PYDOWL_DTYPE_NS}NpArrayNd"

    def __init__(self, val):
        self.val = val


def _np_array_nd_datatype_parser(s: str):
    arr = _bytes_to_ndarray(_b64_to_bytes(s))
    return NpArrayNdDatatype(val=arr)


def _np_array_nd_datatype_unparser(j: NpArrayNdDatatype):
    arr = j.val
    assert arr.ndim != 2, "Use NpArray2dDatatype for 2d np array"
    return _bytes_to_b64(_ndarray_to_bytes(arr))


declare_datatype(
    NpArrayNdDatatype,
    NpArrayNdDatatype.iri,
    _np_array_nd_datatype_parser,
    _np_array_nd_datatype_unparser,
)


# ──────────────────────────────────────────────────────────────────────────
# Custom Datatype: 2-D NumPy matrix (stored as Parquet)
# ──────────────────────────────────────────────────────────────────────────


class NpArray2dDatatype:
    """Efficient 2-D matrix storage via Parquet + Snappy compression."""

    __slots__ = ("val",)
    iri: Final[str] = f"{_PYDOWL_DTYPE_NS}NpArray2d"

    def __init__(self, val):
        self.val = val


def _np_array_2d_datatype_parser(s: str):
    df = pd.read_parquet(io.BytesIO(_b64_to_bytes(s)))
    return NpArray2dDatatype(val=df.values)


def _np_array_2d_datatype_unparser(j: NpArray2dDatatype):
    arr = j.val
    assert arr.ndim == 2
    buf = pd.DataFrame(arr).to_parquet(index=False, compression="snappy")
    return _bytes_to_b64(buf)


declare_datatype(
    NpArray2dDatatype,
    NpArray2dDatatype.iri,
    _np_array_2d_datatype_parser,
    _np_array_2d_datatype_unparser,
)


# ──────────────────────────────────────────────────────────────────────────
# Custom Datatype: Time-zone–aware ISO-8601 datetime
# ──────────────────────────────────────────────────────────────────────────


class DateTimeDatatype:
    """
    A timezone-aware datetime literal.

    Naive datetimes are accepted for compatibility; they are normalised
    to UTC on serialisation so round-trips remain unambiguous.

    In OWL this is represented as an RDF literal with datatype IRI
    ``http://pydowl.org/dtype#DateTime``.
    """

    __slots__ = ("val",)
    iri: Final[str] = f"{_PYDOWL_DTYPE_NS}DateTime"

    def __init__(self, val: datetime.datetime):
        self.val = val


def _date_time_datatype_parser(s: str) -> DateTimeDatatype:
    return DateTimeDatatype(val=datetime.datetime.fromisoformat(s))


def _date_time_datatype_unparser(j: DateTimeDatatype) -> str:
    dt = j.val
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.isoformat()


declare_datatype(
    DateTimeDatatype,
    DateTimeDatatype.iri,
    _date_time_datatype_parser,
    _date_time_datatype_unparser,
)

# ──────────────────────────────────────────────────────────────────────────
# Registry helpers
# ──────────────────────────────────────────────────────────────────────────


CUSTOM_OWL_DATA_TYPES = (
    DateTimeDatatype,
    NpArrayNdDatatype,
    NpArray2dDatatype,
    JsonStringDatatype,
)

# Build parser lookup dict (avoids chained if/elif)
_PARSERS: dict[type, Callable[[str], object]] = {
    DateTimeDatatype: _date_time_datatype_parser,
    NpArrayNdDatatype: _np_array_nd_datatype_parser,
    NpArray2dDatatype: _np_array_2d_datatype_parser,
    JsonStringDatatype: _json_string_datatype_parser,
}


def get_datatype_parser(datatype: DatatypeClass) -> Callable[[str], object]:
    """
    Return the registered *parser* for an OWL datatype class.

    Parameters
    ----------
    datatype:
        The Owlready2 datatype class created via :func:`declare_datatype`.

    Raises
    ------
    DatatypeParserError
        If no parser is registered for ``datatype``.
    """
    try:
        return _PARSERS[datatype]
    except KeyError as exc:  # pragma: no cover - error path tested separately
        raise DatatypeParserError(
            f"No parser registered for datatype {datatype!r}"
        ) from exc


OWLREADY2_BUILTIN_DATATYPE_TABLE = {
    "http://www.w3.org/2001/XMLSchema#integer": int,
    "http://www.w3.org/2001/XMLSchema#decimal": float,
    "http://www.w3.org/2001/XMLSchema#boolean": bool,
    "http://www.w3.org/2001/XMLSchema#base64Binary": bytes,
    "http://www.w3.org/2001/XMLSchema#string": str,
}

CUSTOM_OWL_DATATYPE_TABLE = {dt.iri: dt for dt in CUSTOM_OWL_DATA_TYPES}
