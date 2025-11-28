"""
Public API for the pydowl package.

Most users will interact with:

* :class:`PydOwlClass` and :class:`PydOwlDataClass` as Pydantic bases
  for OWL-aware models.
* :class:`PydOwlRegistry` for optional identity management.
* The datatype and field-type helpers when integrating with custom
  ontologies or storage layers.
"""

from .data_type import (
    JsonStringDatatype,
    NpArrayNdDatatype,
    NpArray2dDatatype,
    DateTimeDatatype,
    CUSTOM_OWL_DATA_TYPES,
    OWLREADY2_BUILTIN_DATATYPE_TABLE,
    CUSTOM_OWL_DATATYPE_TABLE,
    get_datatype_parser,
)
from .field_type import (
    FieldTypeCategory,
    identify_field_type_category,
    validate_pydantic_class,
    get_inner_type,
)
from .base_class import (
    KgBaseModel,
    PydOwlClass,
    PydOwlDataClass,
    PydOwlRegistry,
)
from .version import __version__ as __version__

__all__ = [
    # data_type
    "JsonStringDatatype",
    "NpArrayNdDatatype",
    "NpArray2dDatatype",
    "DateTimeDatatype",
    "CUSTOM_OWL_DATA_TYPES",
    "OWLREADY2_BUILTIN_DATATYPE_TABLE",
    "CUSTOM_OWL_DATATYPE_TABLE",
    "get_datatype_parser",
    # field_type
    "FieldTypeCategory",
    "identify_field_type_category",
    "validate_pydantic_class",
    "get_inner_type",
    # base_class
    "KgBaseModel",
    "PydOwlClass",
    "PydOwlDataClass",
    "PydOwlRegistry",
    # version
    "__version__",
]
