from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pydantic_numpy.typing as pnd
import pytest
from pydantic import Field

from pydowl import PydOwlClass
from pydowl.data_type import (
    JsonStringDatatype,
    NpArrayNdDatatype,
    NpArray2dDatatype,
    DateTimeDatatype,
    CUSTOM_OWL_DATATYPE_TABLE,
    get_datatype_parser,
    DatatypeParserError,
)


class Holder(PydOwlClass):
    arr_nd: Optional[pnd.NpNDArray] = Field(default=None)
    arr_2d: Optional[pnd.Np2DArray] = Field(default=None)
    py_list: list[Any] = Field(default_factory=list)
    py_dict: dict[str, Any] = Field(default_factory=dict)
    ts: Optional[datetime] = Field(default=None)


@pytest.fixture(scope="function")
def h1():
    nd = np.random.rand(2, 3, 4)
    df = np.random.rand(5, 2)
    h1 = Holder(
        identifier="h1",
        arr_nd=nd,
        arr_2d=df,
        py_list=[1, 2, 3],
        py_dict={"a": 1, "b": [1, 2]},
        ts=datetime.now(),
    )
    return h1


def test_pydantic_roundtrip(h1):
    """
    Round trip test for pydantic model dump.
    """
    dumped = h1.model_dump()
    reloaded = Holder.model_validate(dumped)

    assert np.array_equal(reloaded.arr_nd, h1.arr_nd)
    assert np.array_equal(reloaded.arr_2d, h1.arr_2d)
    assert reloaded.py_dict == h1.py_dict
    assert reloaded.ts == h1.ts


def test_owl_roundtrip(test_onto, h1):
    """
    Round trip test for OWL push/pull.
    """
    h1_ind = h1.push_owlready(test_onto, dynamic_tbox=True)
    h1_rec = Holder.pull_owlready(test_onto, h1_ind)
    assert h1_rec.model_dump_json() == h1.model_dump_json()


def test_custom_datatype_iris_and_table():
    """
    E1: Custom datatypes should live under the pydowl namespace and
    appear in CUSTOM_OWL_DATATYPE_TABLE.
    """
    for dt_cls in (
        JsonStringDatatype,
        NpArrayNdDatatype,
        NpArray2dDatatype,
        DateTimeDatatype,
    ):
        iri = dt_cls.iri
        assert iri.startswith("http://pydowl.org/dtype#")
        assert "w3.org/2001/XMLSchema" not in iri
        assert iri in CUSTOM_OWL_DATATYPE_TABLE
        assert CUSTOM_OWL_DATATYPE_TABLE[iri] is dt_cls


def test_get_datatype_parser_known_and_unknown():
    """
    E2: get_datatype_parser should return a callable for known custom
    datatypes and raise a DatatypeParserError for unknown ones.
    """

    # Known type
    parser = get_datatype_parser(JsonStringDatatype)
    assert callable(parser)
    obj = parser('{"foo": 1}')
    assert isinstance(obj, JsonStringDatatype)
    assert obj.val == {"foo": 1}

    # Unknown type
    class Dummy:
        pass

    with pytest.raises(DatatypeParserError):
        # type: ignore[arg-type] – we intentionally pass an unsupported type
        get_datatype_parser(Dummy)
