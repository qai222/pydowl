from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum, IntEnum
from typing import Optional, List, Dict

import pydantic_numpy.typing as pnd
import pytest
from pydantic import Field, ValidationError
from pydantic.fields import FieldInfo

from pydowl import (
    PydOwlClass,
    PydOwlRegistry,
    FieldTypeCategory,
    identify_field_type_category,
)


def reset_registry() -> None:
    PydOwlRegistry.clear()


# ──────────────────────────────────────────────────────────────────────
# Field-type classification (B2 / B3 / B4)
# ──────────────────────────────────────────────────────────────────────


def test_optional_list_is_rejected_when_classified():
    """
    B2: Optional[List[x]] is not supported. We assert that field-type
    classification rejects it.
    """

    class BadOptionalList(PydOwlClass):
        bad: Optional[List[int]] = None

    field_info = BadOptionalList.model_fields["bad"]
    with pytest.raises(TypeError):
        identify_field_type_category(field_info, "bad")


def test_optional_dict_is_rejected_when_classified():
    """
    B2: Optional[Dict[str, x]] is not supported. We assert that
    classification rejects it.
    """

    class BadOptionalDict(PydOwlClass):
        bad: Optional[Dict[str, int]] = None

    field_info = BadOptionalDict.model_fields["bad"]
    with pytest.raises(TypeError):
        identify_field_type_category(field_info, "bad")


def test_py_json_list_and_dict_are_treated_as_blobs():
    """
    B3: Non-Pydowl container types in List/Dict form are treated as
    opaque JSON blobs (PY_JSON).
    """

    class JsonModel(PydOwlClass):
        seq: List[int]
        mapping: Dict[str, int]

    seq_info = JsonModel.model_fields["seq"]
    mapping_info = JsonModel.model_fields["mapping"]

    assert identify_field_type_category(seq_info, "seq") is FieldTypeCategory.PY_JSON
    assert (
        identify_field_type_category(mapping_info, "mapping")
        is FieldTypeCategory.PY_JSON
    )


def test_pydowl_type_field_name_is_reserved():
    """
    B4: The field name 'pydowl_type' is reserved and should not be used
    as a normal field.
    """
    fi = FieldInfo(annotation=str, required=False, default=None)
    with pytest.raises(AssertionError):
        identify_field_type_category(fi, "pydowl_type")


# ──────────────────────────────────────────────────────────────────────
# Enum mapping & category assignment (from old test_fieldtypes.py)
# ──────────────────────────────────────────────────────────────────────


class DummyStrEnum(StrEnum):
    A = "a"


class DummyIntEnum(IntEnum):
    X = 1


class DummySample(PydOwlClass):
    e_str: Optional[DummyStrEnum] = None
    e_int: Optional[DummyIntEnum] = None
    lit: Optional[str] = None
    json_data: Dict[str, int] = Field(default_factory=dict)
    arr_nd: Optional[pnd.NpNDArray] = None
    arr_2d: Optional[pnd.Np2DArray] = None
    ts: Optional[datetime] = None
    children: List[int] = Field(default_factory=list)


def test_category_assignment():
    for name, info in DummySample.model_fields.items():
        cat = identify_field_type_category(info, name)
        if name == "e_str":
            assert cat is FieldTypeCategory.OPTIONAL_STR_ENUM
        elif name == "e_int":
            assert cat is FieldTypeCategory.OPTIONAL_INT_ENUM
        elif name == "lit":
            assert cat is FieldTypeCategory.OPTIONAL_PY_LITERAL
        elif name == "json_data":
            assert cat is FieldTypeCategory.PY_JSON
        elif name == "arr_nd":
            assert cat is FieldTypeCategory.OPTIONAL_NPND_ARRAY
        elif name == "arr_2d":
            assert cat is FieldTypeCategory.OPTIONAL_NP2D_ARRAY
        elif name == "ts":
            assert cat is FieldTypeCategory.OPTIONAL_DATETIME
        elif name == "children":
            assert cat is FieldTypeCategory.PY_JSON


# A small enum-OWL roundtrip test (from old DummyDevice test)


class DummyStatusEnum(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class DummyErrorCodeEnum(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    TIMEOUT = 2


class DummyDevice(PydOwlClass):
    status: Optional[DummyStatusEnum] = Field(default=None)
    error_code: Optional[DummyErrorCodeEnum] = Field(default=None)


def test_enum_field_roundtrip_owl(test_onto):
    device_instance = DummyDevice(status="active", error_code=2)
    device_individual = device_instance.push_owlready(test_onto, dynamic_tbox=True)
    assert device_individual.has_status == "active"
    assert device_individual.has_error_code == 2
    device_instance_rec = DummyDevice.pull_owlready(test_onto, device_individual)
    assert device_instance_rec.model_dump_json() == device_instance.model_dump_json()


# ──────────────────────────────────────────────────────────────────────
# Pydantic behaviour as BaseModel subclass (from test_pydantic.py)
# ──────────────────────────────────────────────────────────────────────


class A(PydOwlClass):
    f1: Optional[int] = Field(default=None)
    f2: List[str] = Field(default_factory=list)


class AA(A):
    f3: Optional[str] = Field(default=None)


class B(PydOwlClass):
    affiliated_A: Optional[A] = Field(default=None)


class BB(B):
    affiliated_A: Optional[AA] = Field(default=None)


A_data1 = dict(f1=42, f2=["a", "d"])
A_data2 = dict(f1=42, f2=["a", "b", "c"], f3="blabla")


@pytest.fixture(scope="module")
def test_a1():
    a1 = A.model_validate(A_data1)
    a1.identifier = "a1"
    return a1


@pytest.fixture(scope="module")
def test_a2():
    a2 = A.model_validate(A_data2)
    a2.identifier = "a2"
    return a2


@pytest.fixture(scope="module")
def test_aa1():
    a1 = AA.model_validate(A_data1)
    a1.identifier = "aa1"
    return a1


@pytest.fixture(scope="module")
def test_aa2():
    a2 = AA.model_validate(A_data2)
    a2.identifier = "aa2"
    return a2


def test_pydantic_behavior1(test_a1, test_a2):
    """
    The `affiliated_A` field of BB is defined as an AA, (AA's parent class) A's instance cannot be used.
    """
    B(affiliated_A=test_a1)
    B(affiliated_A=test_a2)

    with pytest.raises(ValidationError):
        BB(affiliated_A=test_a2)

    with pytest.raises(ValidationError):
        BB(affiliated_A=test_a1)


def test_pydantic_behavior2(test_aa1, test_aa2):
    """
    The `affiliated_A` field of B is defined as an A, (A's child class) AA's instance can be used.
    """
    B(affiliated_A=test_aa1)
    BB(affiliated_A=test_aa1)
    B(affiliated_A=test_aa2)
    BB(affiliated_A=test_aa2)


def test_pydantic_behavior3(test_a1, test_a2, test_aa1, test_aa2):
    """
    make sure the extra field is stored in a2
    """
    assert "f3" in test_a2.__pydantic_extra__
    assert "f3" not in test_a1.__pydantic_extra__
    assert "f3" not in test_aa1.__pydantic_extra__
    assert "f3" not in test_aa2.__pydantic_extra__

    # getattr should succeed on extra field
    assert getattr(test_a2, "f3") == "blabla"


def test_pydantic_behavior4(test_a2):
    """
    make sure the extra field is included in serialization
    """
    assert "f3" in super(PydOwlClass, test_a2).model_dump()
    assert "f3" in test_a2.model_dump()


def test_pydantic_behavior5(test_a1, test_a2, test_aa1, test_aa2):
    """
    If the data is passed as a dictionary, then it will be automatically converted to
    - A if the given model is B
    - AA if the given model is BB
    """
    b1 = B.model_validate(dict(affiliated_A=test_a1.model_dump(), identifier="b1"))
    b2 = B.model_validate(dict(affiliated_A=test_a2.model_dump(), identifier="b2"))
    b3 = B.model_validate(
        dict(affiliated_A=test_aa1.model_dump(), identifier="b3"),
    )
    b4 = B.model_validate(dict(affiliated_A=test_aa2.model_dump(), identifier="b4"))

    # because we allow extra fields, a2 in B would have a 'f3' field, this field also got dumped
    assert getattr(b2.affiliated_A, "f3")
    # but in a1 there is no 'f3' field, so we get AttributeError
    with pytest.raises(AttributeError):
        getattr(b1.affiliated_A, "f3")  # type: ignore[arg-type]

    # of course, they should share the same value
    assert b2.affiliated_A.f3 == b4.affiliated_A.f3

    # by construction aa1 has a None 'f3' field
    assert b3.affiliated_A.f3 is None


def test_pydantic_behavior6(test_a1, test_a2, test_aa1, test_aa2):
    """
    test BB model
    """
    bb1 = BB.model_validate(dict(affiliated_A=test_a1.model_dump(), identifier="bb1"))
    bb2 = BB.model_validate(dict(affiliated_A=test_a2.model_dump(), identifier="bb2"))
    bb3 = BB.model_validate(
        dict(affiliated_A=test_aa1.model_dump(), identifier="bb3"),
    )
    bb4 = BB.model_validate(dict(affiliated_A=test_aa2.model_dump(), identifier="bb4"))

    assert "f3" not in dir(
        bb1.affiliated_A
    )  # test_a1 stayed as an A instance, so it does not have a 'f3' field
    assert bb2.affiliated_A.f3 is not None
    assert bb3.affiliated_A.f3 is None
    assert bb4.affiliated_A.f3 == bb2.affiliated_A.f3


def test_upcast():
    class ParentClass(PydOwlClass):
        pass

    class SubClass1(ParentClass):
        sc1_field1: int
        sc1_field2: str

    class SubClass2(ParentClass):
        sc2_field1: int

    class Container(PydOwlClass):
        f1: ParentClass | None = None
        f2: List[ParentClass] = Field(default_factory=list)

    fq1 = f"{SubClass1.__module__}:{SubClass1.__qualname__}"
    fq2 = f"{SubClass2.__module__}:{SubClass2.__qualname__}"
    input_data = {
        "f1": {
            "pydowl_type": fq1,
            "identifier": "item1",
            "sc1_field1": 1,
            "sc1_field2": "abc",
        },
        "f2": [
            {
                "pydowl_type": fq2,
                "identifier": "item2",
                "sc2_field1": 1,
            },
            {
                "pydowl_type": fq2,  # intentionally wrong class
                "identifier": "item3",
                "sc2_field1": 1,
                "sc1_field1": 1,
                "sc1_field2": "abc",
            },
            {
                "pydowl_type": fq1,
                "identifier": "item1",
                "sc1_field1": 1,
                "sc1_field2": "abc",
            },
        ],
    }

    container = Container.model_validate(input_data)
    assert type(container.f1) is SubClass1
    assert type(container.f2[0]) is SubClass2
    assert type(container.f2[1]) is SubClass2
    assert type(container.f2[2]) is SubClass1


def test_pydowl_type_injection():
    class X(PydOwlClass):
        x: Optional[int] = None

    inst = X(identifier="id1", x=10)

    dumped = inst.model_dump()
    assert "pydowl_type" in dumped
    assert dumped["pydowl_type"].startswith(f"{X.__module__}:{X.__qualname__}")

    # model_dump_json should match the same discriminator
    dumped_json = json.loads(inst.model_dump_json())
    assert dumped_json["pydowl_type"] == dumped["pydowl_type"]


# ──────────────────────────────────────────────────────────────────────
# Core update + registry behaviour (C1–C4 + new register_graph usage)
# ──────────────────────────────────────────────────────────────────────


def test_update_optional_nested_creates_and_attaches_instance():
    reset_registry()

    class Child(PydOwlClass):
        value: Optional[int] = None

    class Parent(PydOwlClass):
        child: Optional[Child] = None

    p = Parent(identifier="parent1", child=None)

    p.update(child={"identifier": "child1", "value": 42})

    assert p.child is not None
    assert isinstance(p.child, Child)
    assert p.child.identifier == "child1"
    assert p.child.value == 42


def test_update_list_nested_merges_by_identifier():
    reset_registry()

    class Item(PydOwlClass):
        value: Optional[int] = None

    class Container(PydOwlClass):
        items: List[Item] = []

    i1 = Item(identifier="i1", value=1)
    c = Container(identifier="c1", items=[i1])

    c.update(
        items=[
            {"identifier": "i1", "value": 2},
            {"identifier": "i2", "value": 3},
        ]
    )

    assert len(c.items) == 2

    updated_i1 = next(i for i in c.items if i.identifier == "i1")
    assert updated_i1 is i1
    assert updated_i1.value == 2

    i2_list = [i for i in c.items if i.identifier == "i2"]
    assert len(i2_list) == 1
    i2 = i2_list[0]
    assert isinstance(i2, Item)
    assert i2.value == 3


def test_from_data_reuses_instance_via_registry():
    reset_registry()

    class Node(PydOwlClass):
        name: Optional[str] = None

    n1 = Node(identifier="n1", name="first")
    PydOwlRegistry.register(n1)

    n2 = Node.from_data({"identifier": "n1", "name": "second"})

    assert n2 is n1
    assert n1.name == "second"


def test_direct_model_validate_creates_new_instance():
    """
    Direct model_validate must *not* reuse the registry – only from_data does.
    """
    reset_registry()

    class Node(PydOwlClass):
        name: Optional[str] = None

    n1 = Node(identifier="n1", name="first")
    PydOwlRegistry.register(n1)

    n2 = Node.model_validate({"identifier": "n1", "name": "second"})

    assert n2 is not n1
    assert n1.name == "first"
    assert n2.name == "second"


def test_registry_register_get_clear_delete():
    reset_registry()

    class Thing(PydOwlClass):
        label: Optional[str] = None

    t = Thing(identifier="t1", label="foo")
    PydOwlRegistry.register(t)

    t2 = PydOwlRegistry.get(Thing, "t1")
    assert t2 is t

    PydOwlRegistry.delete(Thing, "t1")
    assert PydOwlRegistry.get(Thing, "t1") is None

    PydOwlRegistry.clear()
    assert PydOwlRegistry.get(Thing, "t1") is None
