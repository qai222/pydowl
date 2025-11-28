import datetime
import enum
import inspect
import pydantic_numpy.typing as pnd
import strenum
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Type, Dict, List, TypeAlias, get_origin, get_args, Any

"""
pydowl.field_type
=================
Utility helpers that map a *Python type annotation* (on a
``pydantic.BaseModel`` field) to a restricted enumeration
:class:`FieldTypeCategory`.  The mapping drives the Pydantic → OWL /
Mongo adapters in :mod:`pydowl.base_class`.

Supported annotation shapes
---------------------------

The mapping is intentionally conservative. In particular:

* Optional fields **should** use ``typing.Optional[T]`` or
  ``typing.Union[T, None]``.  The PEP 604 form ``T | None`` is **not
  part of the supported API surface** even though, in practice,
  Pydantic / ``typing`` may normalise it to a union that is treated the
  same way. For clarity and portability, prefer ``Optional[T]``.

* JSON-like structures are allowed only in the non-optional forms:
  ``List[x]`` and ``Dict[str, x]``. Optional container forms such as
  ``Optional[List[x]]`` or ``Optional[Dict[str, x]]`` are not supported
  and will raise errors once the field is classified.

* Non-Pydowl list/dict fields are treated as opaque, JSON-serialisable
  blobs (:class:`FieldTypeCategory.PY_JSON`). They are *not* mapped to
  multi-valued OWL data properties.
"""

FieldInfoT: TypeAlias = FieldInfo
InnerType: TypeAlias = type

PYTHON_LITERAL_CLASSES = (
    str,
    float,
    int,
    bytes,
    bool,
)
PND_ANNOTATIONS = frozenset(
    getattr(pnd, name) for name in dir(pnd) if not name.startswith("_")
)
PND_ANNOTATIONS_2D = frozenset(
    getattr(pnd, name)
    for name in dir(pnd)
    if not name.startswith("_") and "2DArray" in name
)


def is_pydantic_model_class(value: Any) -> bool:
    return inspect.isclass(value) and issubclass(value, BaseModel)


def is_optional(field_info: FieldInfoT) -> bool:
    """
    Return ``True`` if the annotation is a union containing ``NoneType``.

    Notes
    -----
    * We intentionally do **not** attempt to distinguish between
      ``Optional[T]`` and the PEP 604 ``T | None`` at runtime, because
      Pydantic / typing normalisation can erase that distinction.
    * The documented and supported style is to use ``Optional[T]`` in
      model definitions.
    """
    args = get_args(field_info.annotation)
    return any(arg is type(None) for arg in args)


def get_inner_type(field_info: FieldInfoT) -> InnerType:
    """
    Extract a useful "inner" type from a field annotation.

    Supported patterns
    ------------------
    * Optional[T] / Union[T, None]:
        return T, but require exactly one non-None member.
    * Simple generics with a single argument (e.g. List[T]):
        return that single argument.

    For anything more complex (e.g. Dict[str, Any]) we return the
    original annotation instead of raising. Callers that *care* about
    the inner type only ever pass in Optional[...] or single-arg
    generics; for everything else this value is ignored.
    """
    annotation = field_info.annotation
    args = get_args(annotation)

    if not args:
        # Non-generic annotation (str, int, ...). For historical uses
        # this should not normally be passed here; keep the previous
        # behaviour of signalling misuse.
        raise AssertionError(f"Cannot extract inner type from {annotation}")

    # Optional[T] / Union[T, None]
    if any(arg is type(None) for arg in args):
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return non_none_types[0]
        # Still strict for weird Optional unions like Union[int, str, None]
        raise AssertionError(f"Unexpected union in {annotation}")

    # Simple generic like List[T], Foo[T], np.typing.NDArray[T]
    if len(args) == 1:
        return args[0]

    # Multi-arg, non-optional generics (Dict[str, Any], Tuple[int, str], ...)
    # Callers that rely on a single inner type (Optionals, List[T]) will
    # never hit this path; for others (e.g. PY_JSON dicts) returning the
    # annotation keeps them working without spurious assertions.
    assert annotation is not None
    return annotation


class FieldTypeCategory(enum.StrEnum):
    """
    Allowed field types in pydowl. Enum docstrings are formatted as:
        <field type = default>. <prop>. <range>
    """

    OPTIONAL_STR_ENUM = "OPTIONAL_STR_ENUM"
    """
    Optional[x] = None, x = StrEnum. data_prop & func_prop. [xsd:string] 
    - Enum values are stored as xsd:string literals in OWL.
    """

    OPTIONAL_INT_ENUM = "OPTIONAL_INT_ENUM"
    """
    Optional[x] = None, x = IntEnum. data_prop & func_prop. [xsd:integer] 
    - Enum values are stored as xsd:integer literals in OWL.
    """

    OPTIONAL_PYD_CLS = "OPTIONAL_PYD_CLS"
    """ 
    Optional[pydantic_class] = None. obj_prop & func_prop. [pydantic_class] 

    - At push time, the referenced object is mapped to an OWL named
      individual of a specific OWL class.
    - At pull time, a named individual is reconstructed as an instance
      of the corresponding Pydantic class.
    """

    OPTIONAL_PY_LITERAL = "OPTIONAL_PY_LITERAL"
    """
    Optional[x] = None, x = str, int, float, bytes, bool. data_prop & func_prop. [x] 

    - Push/pull use Owlready2's built-in literal mappings.
    """

    OPTIONAL_NPND_ARRAY = "OPTIONAL_NPND_ARRAY"
    """
    Optional[x] = None, x = np.array (non-2D). data_prop & func_prop. [NpArrayNdDatatype]
    """

    OPTIONAL_NP2D_ARRAY = "OPTIONAL_NP2D_ARRAY"
    """
    Optional[x] = None, x = 2D np.array. data_prop & func_prop. [NpArray2dDatatype]
    """

    OPTIONAL_DATETIME = "OPTIONAL_DATETIME"
    """
    Optional[x] = None, x = datetime. data_prop & func_prop. [DateTimeDatatype]
    """

    LIST_PYD_CLS = "LIST_PYD_CLS"
    """
    List[pydantic_class] = []. obj_prop. [pydantic_class] 
    """

    PY_JSON = "PY_JSON"
    """
    List[x] = [], x = JSON serializable object. data_prop & func_prop. [JsonStringDatatype] 

    OR

    Dict[str, x] = {}, x = JSON serializable object. data_prop & func_prop. [JsonStringDatatype] 

    Notes
    -----
    * These container fields are treated as **opaque JSON blobs** when
      mapped to OWL and Mongo; they are not expanded into multi-valued
      data properties.
    * Only the *non-optional* forms ``List[x]`` and ``Dict[str, x]`` are
      supported. Optional containers such as ``Optional[List[x]]`` or
      ``Optional[Dict[str, x]]`` are rejected.
    """

    RESERVED = "RESERVED"
    """
    Reserved fields such as ``identifier`` and the discriminator name
    ``pydowl_type``.
    """


def identify_field_type_category(
    field_info: FieldInfo, field_name: str | None = None
) -> FieldTypeCategory:
    """
    Map a Pydantic :class:`FieldInfo` to a :class:`FieldTypeCategory`.

    Parameters
    ----------
    field_info:
        The Pydantic field metadata.
    field_name:
        Optional field name for better error messages and for detecting
        reserved names.

    Raises
    ------
    AssertionError
        If ``field_name == "pydowl_type"`` – this name is reserved for
        the discriminator and must not be used as a normal field.
    TypeError
        If the annotation does not fall into one of the supported
        categories described in the module docstring.
    """
    assert field_name != "pydowl_type", "`pydowl_type` should not appear as a field"
    if field_name and field_name in ["identifier", "pydowl_type"]:
        return FieldTypeCategory.RESERVED

    if is_optional(field_info):
        inner_type = get_inner_type(field_info)
        if is_pydantic_model_class(inner_type):
            return FieldTypeCategory.OPTIONAL_PYD_CLS
        elif inner_type in PYTHON_LITERAL_CLASSES:
            return FieldTypeCategory.OPTIONAL_PY_LITERAL
        elif inner_type == datetime.datetime:
            return FieldTypeCategory.OPTIONAL_DATETIME
        elif inner_type in PND_ANNOTATIONS_2D:
            return FieldTypeCategory.OPTIONAL_NP2D_ARRAY
        elif inner_type in PND_ANNOTATIONS:
            return FieldTypeCategory.OPTIONAL_NPND_ARRAY
        elif isinstance(inner_type, type) and issubclass(inner_type, enum.Enum):
            if issubclass(inner_type, enum.StrEnum) or issubclass(
                inner_type, strenum.StrEnum
            ):
                return FieldTypeCategory.OPTIONAL_STR_ENUM
            elif issubclass(inner_type, enum.IntEnum):
                return FieldTypeCategory.OPTIONAL_INT_ENUM
            raise TypeError(
                f"Enum type {inner_type} is not supported (only StrEnum and IntEnum)."
            )
        else:
            # Optional container types (e.g. Optional[List[x]]) will fall
            # through to this branch and are intentionally rejected.
            raise TypeError(f"An optional type that is not allowed: {field_info}!")
    elif get_origin(field_info.annotation) in (list, List):
        inner_type = get_inner_type(field_info)
        if is_pydantic_model_class(inner_type):
            return FieldTypeCategory.LIST_PYD_CLS
        else:
            return FieldTypeCategory.PY_JSON
    elif get_origin(field_info.annotation) in (dict, Dict):
        key_type, value_type = get_args(field_info.annotation)
        assert key_type is str, "Dict key must be a str"
        return FieldTypeCategory.PY_JSON
    raise TypeError(f"A field type that is not allowed: {field_info}!")


def validate_pydantic_class(pydantic_class: Type[BaseModel]) -> None:
    """
    Validate that *every* field in ``pydantic_class`` is allowed.

    This enforces the field-type whitelist in
    :class:`FieldTypeCategory` and the constraints documented in the
    module-level docstring:

    * Optional-like unions must contain exactly one non-None member.
    * Container fields must be the non-optional ``List[x]`` or
      ``Dict[str, x]`` forms.
    * The discriminator name ``pydowl_type`` is reserved and may not be
      used as a field.

    Raises
    ------
    TypeError
        If any field annotation violates :class:`FieldTypeCategory`.
    AssertionError
        If a field is declared with the reserved name ``pydowl_type``.
    """
    for field_name, field_info in pydantic_class.model_fields.items():
        identify_field_type_category(field_info, field_name)
