import datetime as _dt
import json
import re
import threading
import types
import weakref
from importlib import import_module
from types import ModuleType
from typing import List, Type, TypeVar, Dict, Any, Optional, ClassVar, cast
from uuid import uuid4

import owlready2
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    SerializationInfo,
    ConfigDict,
    model_validator,
    model_serializer,
)

from .data_type import (
    JsonStringDatatype,
    NpArrayNdDatatype,
    DateTimeDatatype,
    CUSTOM_OWL_DATA_TYPES,
    NpArray2dDatatype,
)
from .field_type import (
    identify_field_type_category,
    FieldTypeCategory,
    get_inner_type,
    validate_pydantic_class,
)
from .version import __version__

"""
pydowl.base_class
=================
Core glue between Pydantic models and OWL individuals, plus helpers for:

* Identity & polymorphic discriminators (``identifier`` + ``pydowl_type``)
* OWL push / pull (:class:`PydOwlClass`)
* In-memory instance registry (:class:`PydOwlRegistry`)
* Mongo-style JSON adapters (``to_mongo_docs`` / ``from_mongo_docs``)
* Tree-safe JSON serialisation (``to_tree_dict`` / ``dump_tree_json``)
* Value-object support via :class:`PydOwlDataClass`

The intended high-level flows are:

* Pydantic → Mongo → OWL: ingest validated models into MongoDB as
  node-centric documents, then project into an ontology for reasoning.
* OWL → Pydantic → Mongo: materialise individuals as Pydantic models
  (optionally merging into existing instances via the registry), then
  serialise back into Mongo documents.
"""


# ──────────────────────────────────────────────────────────────────────
# Discriminator resolution
# ──────────────────────────────────────────────────────────────────────


def _all_subclasses(cls: type) -> list[type]:
    """Recursively yield *all* subclasses of ``cls``."""
    subs: list[type] = []
    for sub in cls.__subclasses__():
        subs.append(sub)
        subs.extend(_all_subclasses(sub))
    return subs


def _resolve_class_from_tag(fq_tag: str) -> Type["KgBaseModel"] | None:
    """
    Resolve a fully-qualified discriminator tag into a concrete
    :class:`KgBaseModel` subclass.

    Tag format
    ----------
    ``"<module.path>:<QualName>"``. For nested classes, ``QualName``
    may contain ``"<locals>"`` segments.
    """
    try:
        mod_path, qualname = fq_tag.split(":", 1)
        module: ModuleType = import_module(mod_path)
        obj: Any = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if isinstance(obj, type) and issubclass(obj, KgBaseModel):
            logger.debug("Resolved discriminator '{}' → {}", fq_tag, obj)
            return cast(Type[KgBaseModel], obj)
        else:
            logger.warning(
                "Tag '{}' resolved to {} which is *not* a KgBaseModel subclass",
                fq_tag,
                obj,
            )
    except AttributeError:
        # fall back below
        pass
    except Exception as exc:
        logger.exception("Error while resolving discriminator '{}': {}", fq_tag, exc)

    # Fallback: scan subclasses in case the attribute walk failed (e.g.
    # due to nested class resolution issues).
    try:
        mod_path, qualname = fq_tag.split(":", 1)
        for sub in _all_subclasses(KgBaseModel):
            if sub.__qualname__ == qualname and sub.__module__ == mod_path:
                logger.debug("Fallback resolved discriminator '{}' → {}", fq_tag, sub)
                return cast(Type[KgBaseModel], sub)
        logger.error("Discriminator '{}' could not be resolved", fq_tag)
    except Exception as exc:
        logger.exception("Error while resolving discriminator '{}': {}", fq_tag, exc)
    return None


# ──────────────────────────────────────────────────────────────────────
# KgBaseModel with discriminator injection
# ──────────────────────────────────────────────────────────────────────


class KgBaseModel(BaseModel):
    """
    Base class for every pydowl model.

    Responsibilities
    ----------------
    * **Field-type gatekeeping** – during ``__init_subclass__`` we
      validate field annotations against :mod:`pydowl.field_type`. Any
      unsupported annotation will surface as an error either at import
      time or at first use.

    * **Polymorphic serialisation** – wraps Pydantic's serialiser to
      inject a fully-qualified ``pydowl_type`` discriminator of the form
      ``"<module>:<qualname>"`` on every dump. On deserialisation the
      discriminator is used to up-cast instances to the correct
      subclass via :func:`_resolve_class_from_tag`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "model_fields", None):
            validate_pydantic_class(cls)

    @model_serializer(mode="wrap")
    def inject_type_on_serialization(
        self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo | None = None
    ) -> dict[str, Any]:
        """
        Serialiser hook adding ``pydowl_type`` to the output mapping.

        The value is always computed from the class itself; if a
        ``pydowl_type`` key is already present in the nested output it
        will be overwritten to remain consistent.
        """
        data = nxt(self)
        fq_tag = f"{self.__class__.__module__}:{self.__class__.__qualname__}"
        existing = data.get("pydowl_type")
        if existing is None or existing != fq_tag:
            data["pydowl_type"] = fq_tag
        return data

    @model_validator(mode="wrap")
    @classmethod
    def retrieve_type_on_deserialization(cls, value: Any, handler: Any) -> Any:
        """
        Deserialiser hook using ``pydowl_type`` to select a concrete
        subclass if present.

        If the tag cannot be resolved or is missing, falls back to
        validating against ``cls``.
        """
        if isinstance(value, dict) and (tag := value.pop("pydowl_type", None)):
            resolved = _resolve_class_from_tag(tag)
            if resolved and issubclass(resolved, KgBaseModel):
                return resolved(**value)
        return handler(value)


# ──────────────────────────────────────────────────────────────────────
# Exceptions & property-name helpers
# ──────────────────────────────────────────────────────────────────────


class KgPushError(Exception):
    """Raised when an error occurs while pushing to an OWL ontology."""


class KgPullError(Exception):
    """Raised when an error occurs while pulling / reconstructing data."""


def pyd_attr_to_owl_prop_iri(pyd_attr: str, ontology_iri: str) -> str:
    """
    Map a Pydantic attribute name to an OWL property IRI.

    Convention: ``has_<attribute>`` is appended to the ontology base IRI.
    """
    assert ontology_iri.endswith("#")
    return f"{ontology_iri}has_{pyd_attr}"


def owl_prop_iri_to_pyd_attr(owl_prop: str) -> str:
    """
    Map an OWL property IRI back to a Pydantic attribute name, assuming
    the ``has_<attribute>`` naming convention.
    """
    owl_prop_name = owl_prop.split("#")[-1]
    assert owl_prop_name.startswith("has_"), owl_prop_name
    return owl_prop_name[4:]


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case (very small helper)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# ──────────────────────────────────────────────────────────────────────
# PydOwlClass – OWL + Mongo + tree-safe JSON
# ──────────────────────────────────────────────────────────────────────


class PydOwlClass(KgBaseModel):
    """
    Mixin turning a Pydantic model into an OWL- and Mongo-aware entity.

    Identity
    --------
    * ``identifier`` – used as OWL IRI fragment *and* Mongo ``_id``.
    * ``pydowl_type`` – injected on serialisation, stored in OWL as
      ``has_pydowl_type``, and used to resolve the concrete class.

    Mongo
    -----
    * ``mongo_collection_name`` – class → collection mapping.
    * ``to_mongo_docs`` – node-centric document projection with
      relationship fields as IDs.
    * ``from_mongo_docs`` – reconstruct graph from node-centric docs.

    JSON
    ----
    * ``to_tree_dict`` – nested dict representation of a tree-shaped
      object graph (raises on cycles / shared nodes).
    * ``dump_tree_json`` – JSON wrapper around ``to_tree_dict``.
    """

    identifier: str = Field(default_factory=lambda: str(uuid4()).replace("-", "_"))
    pydowl_version: Optional[str] = Field(default=__version__)

    # Optional override for Mongo collection name
    __mongo_collection__: ClassVar[Optional[str]] = None

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.identifier)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, PydOwlClass) and self.identifier == other.identifier

    # ── F1: class → collection mapping ────────────────────────────────

    @classmethod
    def mongo_collection_name(cls) -> str:
        """
        Return the MongoDB collection name for this class.

        Default: ``snake_case(class name) + "s"``, e.g. ``Person`` →
        ``persons``. Can be overridden by setting
        ``__mongo_collection__`` on the class.
        """
        if cls.__mongo_collection__:
            return cls.__mongo_collection__
        base = _camel_to_snake(cls.__name__)
        if base.endswith("s"):
            return base
        return base + "s"

    # ── OWL pull ──────────────────────────────────────────────────────

    @classmethod
    def pull_owlready(
        cls,
        owlready_onto: owlready2.Ontology,
        owl_ind: owlready2.Thing,
        infer_class: bool = True,
    ) -> "PydOwlClass":
        """
        Materialise *owl_ind* as a (possibly more specific) pydowl object.

        Non-functional data properties are not supported and will raise
        :class:`KgPullError`.
        """

        def _pull_owlready(
            _individual: owlready2.Thing,
            _owlready_onto: owlready2.Ontology,
            _default_cls: Type["PydOwlClass"],
        ) -> "PydOwlClass":
            target_cls: Type["PydOwlClass"] = _default_cls
            try:
                tag_literal = _individual.has_pydowl_type  # functional → single value
                resolved_cls = _resolve_class_from_tag(tag_literal)
                if resolved_cls and issubclass(resolved_cls, PydOwlClass):
                    target_cls = cast(Type[PydOwlClass], resolved_cls)
            except (AttributeError, IndexError):
                pass

            data: dict[str, Any] = {"identifier": _individual.name}

            for prop in _individual.get_properties():
                pyd_attr_name = owl_prop_iri_to_pyd_attr(prop.iri)
                if isinstance(prop, owlready2.DataPropertyClass):
                    if owlready2.FunctionalProperty not in prop.is_a:
                        raise KgPullError(
                            f"Data property {prop} on {_individual} is not functional; "
                            "pydowl only supports functional data properties."
                        )
                    val = getattr(_individual, prop.name)
                    if isinstance(val, CUSTOM_OWL_DATA_TYPES):
                        val = val.val
                    data[pyd_attr_name] = val
                elif isinstance(prop, owlready2.ObjectPropertyClass):
                    related = prop[_individual]
                    if len(related) == 0:
                        data[pyd_attr_name] = None
                    elif owlready2.FunctionalProperty in prop.is_a:
                        data[pyd_attr_name] = _pull_owlready(
                            related[0], _owlready_onto, _default_cls
                        )
                    else:
                        data[pyd_attr_name] = [
                            _pull_owlready(obj, _owlready_onto, _default_cls)
                            for obj in related
                        ]
            return target_cls.from_data(data)

        return _pull_owlready(owl_ind, owlready_onto, cls)

    # ── OWL push (with cycle handling) ────────────────────────────────

    def push_owlready(
        self,
        owlready_onto: owlready2.Ontology,
        dynamic_tbox: bool = False,
        merge: bool = True,
    ) -> owlready2.Thing:
        """
        Push this instance to an OWL ontology as an individual.

        If an individual with the same IRI already exists, its properties
        are updated. Cycles in the object graph are handled via a
        visited-map keyed by ``(cls, identifier)``.

        Parameters
        ----------
        owlready_onto:
            Target ontology.
        dynamic_tbox:
            If True, missing classes/properties are created on the fly.
        merge:
            If True (default), list-valued fields are *merged* with any
            existing OWL values (union-sert, current behavior).
            If False, list-valued fields in OWL are *replaced* so that
            the OWL list matches this model exactly.
        """
        if self.__pydantic_extra__:
            raise ValueError(
                f"Unmodelled fields {list(self.__pydantic_extra__)} "
                "must be added to the schema before pushing to KG"
            )
        return PydOwlClass._push_owlready(
            self,
            owlready_onto,
            dynamic_tbox,
            merge=merge,
        )

    @staticmethod
    def _push_owlready(
        model_instance: "PydOwlClass",
        owlready_onto: owlready2.Ontology,
        dynamic_creation: bool,
        merge: bool = True,
        _visited: Optional[
            Dict[tuple[Type["PydOwlClass"], str], owlready2.Thing]
        ] = None,
    ) -> owlready2.Thing:
        if _visited is None:
            _visited = {}

        cls_type = type(model_instance)
        visit_key = (cls_type, model_instance.identifier)
        if visit_key in _visited:
            return _visited[visit_key]

        cls_name = cls_type.__name__
        logger.debug(
            f"Pushing pydantic class '{cls_name}' with identifier '{model_instance.identifier}'",
        )

        with owlready_onto:
            # Phase 1 – ensure OWL class exists
            owl_class = owlready_onto.search_one(
                iri=f"{owlready_onto.base_iri}{cls_name}"
            )

            if owl_class is None and dynamic_creation:
                pydantic_class = cls_type
                owl_base_classes: list[type] = []

                for base_pyd in pydantic_class.__bases__:
                    if (
                        base_pyd is BaseModel
                        or base_pyd is KgBaseModel
                        or base_pyd is PydOwlClass
                    ):
                        continue  # Skip framework bases

                    base_cls_name = base_pyd.__name__
                    base_owl_class = owlready_onto.search_one(
                        iri=f"{owlready_onto.base_iri}{base_cls_name}"
                    )

                    if not base_owl_class:
                        try:
                            base_instance = base_pyd()
                        except TypeError:
                            base_owl_class = owlready2.Thing
                        else:
                            dummy_entity = PydOwlClass._push_owlready(
                                base_instance,
                                owlready_onto,
                                dynamic_creation,
                                merge=merge,
                                _visited=_visited,
                            )
                            base_owl_class = dummy_entity.__class__
                            owlready2.destroy_entity(dummy_entity)
                            logger.debug(
                                "Destroy dummy entity: '{}' '{}'",
                                base_owl_class,
                                dummy_entity.iri,
                            )
                    owl_base_classes.append(base_owl_class)

                if owl_base_classes:
                    owl_class = types.new_class(cls_name, bases=tuple(owl_base_classes))
                else:
                    owl_class = types.new_class(cls_name, bases=(owlready2.Thing,))

            elif owl_class is None and not dynamic_creation:
                raise KgPushError(
                    f"OWL class not found for {cls_name} and dynamic creation is disabled!"
                )

            # Phase 2 – ensure individual exists
            identifier = model_instance.identifier
            individual_iri = f"{owlready_onto.base_iri}{identifier}"

            individual = owlready_onto.search_one(iri=individual_iri)
            if individual is None:
                individual = owl_class(iri=individual_iri)
                logger.debug(f"creating individual: {individual_iri}")
            else:
                if owl_class not in individual.is_a:
                    individual.is_a.append(owl_class)

            # Record in visited map before recursing
            _visited[visit_key] = individual

            # Ensure has_pydowl_type exists and set it
            prop_tag_name = "has_pydowl_type"
            prop_tag_iri = f"{owlready_onto.base_iri}{prop_tag_name}"
            tag_prop = owlready_onto.search_one(iri=prop_tag_iri)
            if not tag_prop:
                with owlready_onto:

                    class has_pydowl_type(
                        owlready2.DataProperty, owlready2.FunctionalProperty
                    ):
                        range = [str]

            fq_tag = f"{cls_type.__module__}:{cls_type.__qualname__}"
            setattr(individual, prop_tag_name, fq_tag)

            # Phase 3 – set properties
            for field_name, field_info in cls_type.model_fields.items():
                if field_name == "identifier":
                    continue

                field_value = getattr(model_instance, field_name)
                if field_value is None:
                    continue

                field_category = identify_field_type_category(field_info, field_name)

                prop_iri = pyd_attr_to_owl_prop_iri(
                    field_name, ontology_iri=owlready_onto.base_iri
                )
                prop_name = prop_iri.split("#")[-1]
                prop = owlready_onto.search_one(iri=prop_iri)

                if not prop and not dynamic_creation:
                    raise KgPushError(
                        f"OWL object/data property not found for {field_name} and dynamic creation is disabled!"
                    )
                elif not prop and dynamic_creation:
                    if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                        prop = types.new_class(
                            prop_name,
                            bases=(
                                owlready2.ObjectProperty,
                                owlready2.FunctionalProperty,
                            ),
                        )
                    elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                        prop = types.new_class(
                            prop_name, bases=(owlready2.ObjectProperty,)
                        )
                    else:
                        prop = types.new_class(
                            prop_name,
                            bases=(
                                owlready2.DataProperty,
                                owlready2.FunctionalProperty,
                            ),
                        )

                # NOTE: we deliberately do not manage domain/range axioms.

                if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    related_individual = PydOwlClass._push_owlready(
                        field_value,
                        owlready_onto,
                        dynamic_creation=dynamic_creation,
                        merge=merge,
                        _visited=_visited,
                    )
                    setattr(individual, prop.name, related_individual)

                elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                    if not merge:
                        # Replace semantics: OWL list becomes exactly this Python list
                        related_individuals: list[owlready2.Thing] = []
                        for item in field_value:
                            related_individual = PydOwlClass._push_owlready(
                                item,
                                owlready_onto,
                                dynamic_creation=dynamic_creation,
                                merge=merge,
                                _visited=_visited,
                            )
                            related_individuals.append(related_individual)
                        # Assign the full list at once; owlready2 will overwrite previous values.
                        prop[individual] = related_individuals
                    else:
                        # Existing behavior: append (union-sert)
                        for item in field_value:
                            related_individual = PydOwlClass._push_owlready(
                                item,
                                owlready_onto,
                                dynamic_creation=dynamic_creation,
                                merge=merge,
                                _visited=_visited,
                            )
                            prop[individual].append(related_individual)

                elif field_category == FieldTypeCategory.OPTIONAL_PY_LITERAL:
                    setattr(individual, prop.name, field_value)

                elif field_category == FieldTypeCategory.PY_JSON:
                    setattr(individual, prop.name, JsonStringDatatype(val=field_value))

                elif field_category == FieldTypeCategory.OPTIONAL_NP2D_ARRAY:
                    setattr(individual, prop.name, NpArray2dDatatype(val=field_value))

                elif field_category == FieldTypeCategory.OPTIONAL_NPND_ARRAY:
                    setattr(individual, prop.name, NpArrayNdDatatype(val=field_value))

                elif field_category == FieldTypeCategory.OPTIONAL_DATETIME:
                    setattr(individual, prop.name, DateTimeDatatype(val=field_value))

                elif field_category == FieldTypeCategory.OPTIONAL_STR_ENUM:
                    # OWL treats StrEnum as xsd:string
                    setattr(individual, prop.name, field_value.value)

                elif field_category == FieldTypeCategory.OPTIONAL_INT_ENUM:
                    # OWL treats IntEnum as xsd:integer
                    setattr(individual, prop.name, field_value.value)

                else:
                    raise TypeError(f"Unknown field category: {field_category}")

            logger.debug(f"finished pushing '{individual.iri}' as '{owl_class}'")
            return individual

    # ── Pydantic config ───────────────────────────────────────────────

    model_config = ConfigDict(
        extra="allow",
        # NOTE: this is important in pulling from KG -- the pulled data can have extra fields.
        # 'forbid' would error out when extra fields are pulled, 'ignore' would not set extra fields without error out
        # both of them defeat the purpose of having connected data
        # we choose 'allow' so extra fields are captured
        # NOTE: setting this to 'allow' should not interfere with `push_owlready` for dynamic creating data/object
        # properties -- there we use `model_fields` to get candidate properties, which has nothing to do with an
        # instance with extra fields.
        # NOTE: because we use `model_fields`, extra fields would NOT be pushed to owlready.
        # TODO ideally we set this to 'allow' only when pulling from KG, and to 'ignore' or 'forbid' when we want to be
        #  rigorous with data structure
        validate_assignment=True,
        # ^ setting this True breaks the registry causing stack overflow.
        # The workaround is to use `self.__dict__[key] = value` in registry-related updates to bypass validation there.
        # populate_by_name=True,   # enable if you choose to use field aliases later
    )

    # ── Registry-aware construction ───────────────────────────────────

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "PydOwlClass":
        """
        Construct or reuse an instance based on ``identifier`` and type.

        If an instance with the same ``(cls, identifier)`` exists in
        :class:`PydOwlRegistry`, it is updated and returned; otherwise a
        new instance is created, registered and returned.
        """
        unique_id = data.get("identifier")
        if unique_id is not None:
            existing = PydOwlRegistry.get(cast(type["PydOwlClass"], cls), unique_id)
            if existing:
                logger.debug(
                    "found an existing instance in the registry: '%s', it will be updated",
                    existing.identifier,
                )
                existing.update(**data)
                return existing
        instance = cls.model_validate(data)
        PydOwlRegistry.register(instance)
        return instance

    # ── Registry-friendly merge ───────────────────────────────────────

    def update(self, **data: Any) -> None:
        """
        Merge *data* into the current instance, honouring nested
        PydOwlClass relationships and lists.

        Assignments bypass Pydantic validation by writing directly into
        ``__dict__`` to avoid recursion issues with
        ``validate_assignment=True``.
        """
        for key, value in data.items():
            if key == "identifier":
                continue
            if key in type(self).model_fields:
                field_value_existing = getattr(self, key)
                field_value_incoming = value
                field_info = type(self).model_fields[key]
                field_category = identify_field_type_category(field_info, key)
                field_type = get_inner_type(field_info)
                field_type = cast(Type[PydOwlClass], field_type)

                if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    if field_value_incoming is None:
                        self.__dict__[key] = field_value_incoming
                    elif isinstance(field_value_incoming, dict):
                        if field_value_existing is None:
                            nested = field_type(
                                identifier=field_value_incoming["identifier"]
                            )
                            nested.update(**field_value_incoming)
                            self.__dict__[key] = nested
                        else:
                            assert isinstance(field_value_existing, PydOwlClass)
                            field_value_existing.update(**field_value_incoming)
                    elif isinstance(field_value_incoming, PydOwlClass):
                        if (
                            field_value_existing is None
                            or not isinstance(field_value_existing, PydOwlClass)
                            or field_value_existing.identifier
                            != field_value_incoming.identifier
                        ):
                            self.__dict__[key] = field_value_incoming
                        else:
                            field_value_existing.update(
                                **field_value_incoming.model_dump()
                            )
                    else:
                        raise TypeError(
                            f"Unexpected incoming type for field '{key}': "
                            f"{type(field_value_incoming)}"
                        )

                elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                    assert isinstance(field_value_incoming, list)
                    if field_value_existing is None:
                        field_value_existing_list: list[PydOwlClass] = []
                    else:
                        assert isinstance(field_value_existing, list)
                        field_value_existing_list = field_value_existing

                    if not field_value_incoming:
                        self.__dict__[key] = field_value_incoming
                    else:
                        id2existing: Dict[str, PydOwlClass] = {
                            i.identifier: i for i in field_value_existing_list
                        }
                        updated_list: list[PydOwlClass] = []

                        for item in field_value_incoming:
                            if isinstance(item, PydOwlClass):
                                incoming_obj = item
                            elif isinstance(item, dict):
                                incoming_id = item["identifier"]
                                existing_obj = id2existing.get(incoming_id)
                                if existing_obj is not None:
                                    existing_obj.update(**item)
                                    incoming_obj = existing_obj
                                else:
                                    incoming_obj = field_type(identifier=incoming_id)
                                    incoming_obj.update(**item)
                            else:
                                raise TypeError(
                                    f"Unexpected list element type for field '{key}': "
                                    f"{type(item)}"
                                )

                            updated_list.append(incoming_obj)

                        self.__dict__[key] = updated_list

                else:
                    self.__dict__[key] = field_value_incoming
            else:
                logger.warning(
                    "trying to update a field '%s' which is not defined in '%s', "
                    "this field is ignored!",
                    key,
                    self.__class__,
                )

    # ── F2: Pydantic → Mongo node docs ───────────────────────────────

    def to_mongo_docs(self) -> List[Dict[str, Any]]:
        """
        Serialise this object graph into a list of MongoDB documents.

        Each document is node-centric:

        * ``_id`` – the ``identifier`` of the instance
        * ``pydowl_type`` – fully-qualified class path
        * scalar fields – JSON-compatible values
        * relationship fields – identifiers (or lists of identifiers)
          of related :class:`PydOwlClass` instances
        """
        docs: Dict[tuple[Type["PydOwlClass"], str], Dict[str, Any]] = {}
        stack: list[PydOwlClass] = [self]

        while stack:
            inst = stack.pop()
            cls_type = type(inst)
            key = (cls_type, inst.identifier)
            if key in docs:
                continue

            fq_tag = f"{cls_type.__module__}:{cls_type.__qualname__}"
            doc: Dict[str, Any] = {
                "_id": inst.identifier,
                "pydowl_type": fq_tag,
            }

            for field_name, field_info in cls_type.model_fields.items():
                if field_name in ("identifier", "pydowl_version"):
                    continue
                value = getattr(inst, field_name)
                if value is None:
                    continue

                field_category = identify_field_type_category(field_info, field_name)

                if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    assert isinstance(value, PydOwlClass)
                    doc[field_name] = value.identifier
                    stack.append(value)

                elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                    assert isinstance(value, list)
                    ids: list[str] = []
                    for child in value:
                        assert isinstance(child, PydOwlClass)
                        ids.append(child.identifier)
                        stack.append(child)
                    doc[field_name] = ids

                elif field_category == FieldTypeCategory.OPTIONAL_DATETIME:
                    assert isinstance(value, _dt.datetime)
                    doc[field_name] = value.isoformat()

                else:
                    # All other scalar / JSON-like categories are stored
                    # as-is; callers are responsible for ensuring JSON
                    # serialisability if needed.
                    doc[field_name] = value

            docs[key] = doc

        return list(docs.values())

    # ── F3: Mongo docs → Pydantic graph ──────────────────────────────

    @classmethod
    def from_mongo_docs(cls, root_id: str, docs: List[Dict[str, Any]]) -> "PydOwlClass":
        """
        Reconstruct an object graph from a set of MongoDB documents.

        Parameters
        ----------
        root_id:
            Identifier (``_id``) of the root node.
        docs:
            Iterable of documents as produced by :meth:`to_mongo_docs`.

        Returns
        -------
        PydOwlClass
            The root instance. Related instances are constructed as
            needed and registered in :class:`PydOwlRegistry`.

        Raises
        ------
        KgPullError
            If a referenced identifier cannot be found in ``docs``.
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            if "_id" not in d:
                raise KgPullError("Mongo document missing '_id' field")
            by_id[d["_id"]] = d

        visited: Dict[str, PydOwlClass] = {}

        def build(node_id: str) -> PydOwlClass:
            if node_id in visited:
                return visited[node_id]

            doc = by_id.get(node_id)
            if doc is None:
                raise KgPullError(f"Referenced identifier '{node_id}' not found")

            tag = doc.get("pydowl_type")
            target_cls: Type[PydOwlClass]
            if tag:
                resolved = _resolve_class_from_tag(tag)
                if resolved and issubclass(resolved, PydOwlClass):
                    target_cls = cast(Type[PydOwlClass], resolved)
                else:
                    target_cls = cls
            else:
                target_cls = cls

            update_data: Dict[str, Any] = {}
            # Special case: value objects (PydOwlDataClass) must be
            # constructed in one shot so their identity fields are never
            # mutated after initialisation. We therefore build the field
            # data first and call from_data(update_data) directly, rather
            # than using the placeholder+update path.
            if issubclass(target_cls, PydOwlDataClass):
                for field_name, field_info in target_cls.model_fields.items():
                    if field_name in ("identifier", "pydowl_version"):
                        continue
                    if field_name not in doc:
                        continue
                    stored_value = doc[field_name]
                    field_category = identify_field_type_category(
                        field_info, field_name
                    )

                    if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                        if stored_value is None:
                            update_data[field_name] = None
                        else:
                            assert isinstance(stored_value, str)
                            update_data[field_name] = build(stored_value)

                    elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                        if stored_value is None:
                            update_data[field_name] = []
                        else:
                            assert isinstance(stored_value, list)
                            update_data[field_name] = [
                                build(child_id) for child_id in stored_value
                            ]

                    elif field_category == FieldTypeCategory.OPTIONAL_DATETIME:
                        if stored_value is None:
                            update_data[field_name] = None
                        else:
                            assert isinstance(stored_value, str)
                            update_data[field_name] = _dt.datetime.fromisoformat(
                                stored_value
                            )

                    else:
                        update_data[field_name] = stored_value

                inst = target_cls.from_data(update_data)
                visited[node_id] = inst
                return inst

            # Normal entity path (cycle-safe):
            # First create a placeholder with only identifier so cycles
            # can be resolved; from_data will reuse an existing instance
            # if present in the registry.
            placeholder = target_cls.from_data({"identifier": node_id})
            visited[node_id] = placeholder

            for field_name, field_info in target_cls.model_fields.items():
                if field_name in ("identifier", "pydowl_version"):
                    continue
                if field_name not in doc:
                    continue

                stored_value = doc[field_name]
                field_category = identify_field_type_category(field_info, field_name)

                if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    if stored_value is None:
                        update_data[field_name] = None
                    else:
                        assert isinstance(stored_value, str)
                        update_data[field_name] = build(stored_value)

                elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                    if stored_value is None:
                        update_data[field_name] = []
                    else:
                        assert isinstance(stored_value, list)
                        update_data[field_name] = [
                            build(child_id) for child_id in stored_value
                        ]

                elif field_category == FieldTypeCategory.OPTIONAL_DATETIME:
                    if stored_value is None:
                        update_data[field_name] = None
                    else:
                        assert isinstance(stored_value, str)
                        update_data[field_name] = _dt.datetime.fromisoformat(
                            stored_value
                        )

                else:
                    # All other scalar / JSON-like categories are stored as-is.
                    update_data[field_name] = stored_value

            placeholder.update(**update_data)
            return placeholder

        return build(root_id)

    # ── G1/G2: tree-safe JSON serialisation ──────────────────────────

    def to_tree_dict(self) -> Dict[str, Any]:
        """
        Serialise this instance and its reachable :class:`PydOwlClass`
        children into a nested dict, assuming the graph is a tree.

        If the graph contains cycles or shared nodes (a DAG), a
        :class:`ValueError` is raised.
        """
        visited: set[tuple[Type["PydOwlClass"], str]] = set()

        def walk(node: "PydOwlClass") -> Dict[str, Any]:
            key = (type(node), node.identifier)
            if key in visited:
                raise ValueError(
                    "Object graph is not a tree (cycle or shared node detected); "
                    "use a graph-aware serialiser instead."
                )
            visited.add(key)

            cls_type = type(node)
            out: Dict[str, Any] = {
                "identifier": node.identifier,
                "pydowl_type": f"{cls_type.__module__}:{cls_type.__qualname__}",
            }

            for field_name, field_info in cls_type.model_fields.items():
                if field_name in ("identifier", "pydowl_version"):
                    continue
                value = getattr(node, field_name)
                if value is None:
                    continue

                field_category = identify_field_type_category(field_info, field_name)

                if field_category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    assert isinstance(value, PydOwlClass)
                    out[field_name] = walk(value)

                elif field_category == FieldTypeCategory.LIST_PYD_CLS:
                    assert isinstance(value, list)
                    out[field_name] = [walk(child) for child in value]

                else:
                    out[field_name] = value

            return out

        return walk(self)

    def dump_tree_json(self, **json_kwargs: Any) -> str:
        """
        Convenience wrapper around :meth:`to_tree_dict` that returns a
        JSON string.

        Non-JSON-native values (e.g. datetimes) are stringified via
        ``default=str``.  Callers can override or extend ``json_kwargs``
        as needed.
        """
        data = self.to_tree_dict()
        kwargs = {"default": str}
        kwargs.update(json_kwargs)
        return json.dumps(data, **kwargs)  # type:ignore


# ──────────────────────────────────────────────────────────────────────
# PydOwlDataClass – value-object variant
# ──────────────────────────────────────────────────────────────────────


class PydOwlDataClass(PydOwlClass):
    """
    Value-object variant of :class:`PydOwlClass`.

    Identity and immutability
    -------------------------
    * Identity is derived from a subset of fields (``__id_fields__``) or,
      by default, all fields except ``identifier`` and ``pydowl_version``.
    * A stable identifier is computed as::

          "<pydowl_type>:<hash16>"

      where ``pydowl_type`` is the fully-qualified class path
      (``"<module>:<qualname>"``) and ``hash16`` is the first 16 hex
      characters of a SHA-256 digest over the class + id-field values.
    * Fields participating in identity must **never change** after
      initialisation. Attempts to change them, whether via direct
      assignment or via :meth:`update`, raise :class:`TypeError`.

    Notes
    -----
    * For objects reconstructed from storage (Mongo/OWL/SPARQL), an
      explicit ``identifier`` field is respected and **not** recomputed;
      we assume the persisted ID was produced by the same hashing
      scheme.
    * Merges via :meth:`update` are allowed only if they do **not**
      change the identity fields. Non-identity fields remain mutable.
      If you need a different logical value, construct a new instance.
    """

    __id_fields__: ClassVar[Optional[tuple[str, ...]]] = None

    @classmethod
    def _id_field_names(cls) -> tuple[str, ...]:
        """
        Return the tuple of field names that participate in the identity
        hash for this dataclass.

        By default this is every model field except ``identifier`` and
        ``pydowl_version``; subclasses may override ``__id_fields__`` to
        control this explicitly.
        """
        if cls.__id_fields__ is not None:
            return cls.__id_fields__
        return tuple(
            name
            for name in cls.model_fields
            if name not in ("identifier", "pydowl_version")
        )

    @model_validator(mode="after")
    def _set_stable_identifier(self) -> "PydOwlDataClass":
        """
        After validation, compute a stable identifier if one was not
        explicitly provided in the input data, and mark the identity
        fields as frozen.
        """
        fields_set: set = getattr(self, "__pydantic_fields_set__", set())
        id_fields = type(self)._id_field_names()

        # If identifier was explicitly provided, respect it and do not recompute.
        if "identifier" not in fields_set:
            stable_id = self._compute_stable_identifier(id_fields)
            object.__setattr__(self, "identifier", stable_id)

        # Mark as frozen for __setattr__ checks
        object.__setattr__(self, "_pydowl_dataclass_frozen", True)
        return self

    def _compute_stable_stem(self, id_fields: tuple[str, ...]) -> str:
        """
        Build a canonical JSON string representing the identity payload.
        """
        cls = type(self)
        fq_tag = f"{cls.__module__}:{cls.__qualname__}"
        payload: dict[str, Any] = {"class": fq_tag, "fields": {}}
        fields_dict: dict[str, Any] = payload["fields"]
        for name in id_fields:
            val = getattr(self, name)
            if isinstance(val, PydOwlClass):
                fields_dict[name] = {
                    "pydowl_type": f"{type(val).__module__}:{type(val).__qualname__}",
                    "identifier": val.identifier,
                }
            elif isinstance(val, _dt.datetime):
                fields_dict[name] = val.isoformat()
            else:
                fields_dict[name] = val
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return s

    def _compute_stable_identifier(self, id_fields: tuple[str, ...]) -> str:
        """
        Compute ``<pydowl_type>:<hash16>`` from the identity payload.
        """
        cls = type(self)
        pydowl_type = f"{cls.__module__}:{cls.__qualname__}"
        stem = self._compute_stable_stem(id_fields)
        import hashlib

        digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()
        short_hash = digest[:16]  # 64 bits of hash; short but robust
        return f"{pydowl_type}:{short_hash}"

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Enforce immutability for identity fields after initialisation.

        Direct assignment to any identity field (as defined by
        :meth:`_id_field_names`) after the dataclass has been validated
        will raise :class:`TypeError`. Non-identity fields remain
        mutable.
        """
        if getattr(self, "_pydowl_dataclass_frozen", False):
            id_fields = type(self)._id_field_names()
            if name in id_fields and hasattr(self, name):
                raise TypeError(
                    f"Field '{name}' is part of the identity of {type(self).__name__} "
                    "and cannot be modified after initialisation. "
                    "Create a new instance instead."
                )
        super().__setattr__(name, value)

    def update(self, **data: Any) -> None:
        """
        Merge *data* into the current instance, enforcing identity
        immutability.

        Any attempt to change one of the identity fields (as returned by
        :meth:`_id_field_names`) results in :class:`TypeError`. Non-
        identity fields are merged using the normal :class:`PydOwlClass`
        :meth:`update` semantics.
        """
        id_fields = type(self)._id_field_names()
        for name in id_fields:
            if name in data:
                incoming = data[name]
                current = getattr(self, name)
                # Only allow updates that do not change the identity
                # value. For complex types this relies on Python's
                # equality semantics; if in doubt, this will err on the
                # side of raising.
                if incoming != current:
                    raise TypeError(
                        f"Cannot update identity field '{name}' on {type(self).__name__}; "
                        "value objects are immutable. Create a new instance instead."
                    )

        # Delegate to base-class update for non-identity fields
        super().update(**data)


# ──────────────────────────────────────────────────────────────────────
# Instance registry
# ──────────────────────────────────────────────────────────────────────


T = TypeVar("T", bound="PydOwlClass")


class PydOwlRegistry:
    """
    Global identity map for :class:`PydOwlClass` instances.

    Keys
    ----
    The registry is keyed by the pair ``(model_type, identifier)``,
    where:

    * ``model_type`` – concrete subclass of :class:`PydOwlClass`
    * ``identifier`` – the instance's ``identifier`` field

    Values are weak references, so instances can be garbage collected
    when no longer used elsewhere.

    Implicit usage
    --------------
    The registry is used implicitly by :meth:`PydOwlClass.from_data`,
    which is in turn called by higher-level helpers:

    * :meth:`PydOwlClass.from_mongo_docs`
    * :meth:`PydOwlClass.pull_owlready`

    These helpers always construct new objects *via* ``from_data``, so:

    * If an instance with a given ``(cls, identifier)`` is already in
      the registry, it is **reused** and updated in-place.
    * Otherwise a **new** instance is created, registered, and returned.

    This gives you:

    * **Cross-call identity reuse** – e.g. calling
      ``Person.from_mongo_docs("alice", ...)`` multiple times will
      return the *same* `Person` instance as long as it remains
      registered and alive.
    * **Merging overlapping subgraphs** – if two separate loads touch
      the same underlying node (same class + identifier), they will
      converge to the same in-memory object instead of diverging into
      duplicates.

    When to call ``register`` / ``register_graph``
    ----------------------------------------------
    In many pipelines you do *not* need to call the registry manually:

    * A single call to :meth:`from_mongo_docs` or
      :meth:`pull_owlready` will build a coherent object graph using
      its own local visited-map, regardless of the global registry.
    * Those helpers also *populate* the registry as they create
      instances, so subsequent loads can reuse them.

    Explicit registration is useful when:

    * You construct objects **manually** (e.g. from an API payload or
      business logic) and later want data pulled from Mongo / OWL to
      **merge into those exact instances** rather than creating new
      ones.  In that case, register the instances (or the whole graph)
      *before* calling :meth:`from_mongo_docs` / :meth:`pull_owlready`.

    For example::

        alice = Person(identifier="alice", name="Alice")
        PydOwlRegistry.register(alice)
        # later...
        alice2 = Person.from_mongo_docs("alice", docs)
        assert alice2 is alice  # registry reuse

    * You want to ensure that overlapping subgraphs loaded in different
      places share a common object for the same node.  For a manually
      constructed graph, :meth:`register_graph` can be used to seed the
      registry with all reachable nodes from a given root.

    See :meth:`register_graph` for a convenience API that registers an
    entire PydOwl graph rather than a single instance.
    """

    _lock = threading.Lock()
    _registry: Dict[Type[PydOwlClass], Dict[str, weakref.ref[PydOwlClass]]] = {}

    @classmethod
    def register(cls, instance: PydOwlClass) -> None:
        """
        Register a *single* instance under its concrete type and identifier.

        This does **not** traverse the object graph; only ``instance``
        itself is recorded.  To register an entire graph of connected
        :class:`PydOwlClass` objects starting from a root, use
        :meth:`register_graph`.
        """
        with cls._lock:
            model_type = type(instance)
            if model_type not in cls._registry:
                cls._registry[model_type] = {}
            cls._registry[model_type][instance.identifier] = weakref.ref(instance)

    @classmethod
    def register_graph(cls, root: PydOwlClass) -> None:
        """
        Recursively register *root* and all reachable :class:`PydOwlClass`
        instances in its object graph.

        Traversal follows only fields that are modelled as:

        * ``OPTIONAL_PYD_CLS`` – a single optional :class:`PydOwlClass`
          field
        * ``LIST_PYD_CLS`` – a list of :class:`PydOwlClass` instances

        Other fields (scalars, JSON blobs, NumPy arrays, etc.) are
        ignored.  Cycles are handled via a visited-set keyed by
        ``(cls, identifier)``.

        Use this when:

        * You manually construct an object graph in Python and want
          subsequent loads from Mongo / OWL to **reuse and update** the
          same in-memory instances instead of creating duplicates.

        Example
        -------

        >>> a = Bar(identifier="bar", bar_field="bar_field")
        >>> a1 = Bar(identifier="a1", bar_field="bar_field")
        >>> b = Foo(identifier="x", bar=a, bars=[a1, a])
        >>> PydOwlRegistry.register_graph(b)
        >>> # later...
        >>> b2 = Foo.pull_owlready(onto, some_individual)
        >>> assert b2 is b
        >>> assert b2.bar is a
        >>> assert b2.bars[0] is a1

        This mirrors "identity maps" in ORMs: there is at most one live
        Python object per underlying node (type + identifier) within a
        process, assuming all relevant instances have been registered.
        """
        stack: list[PydOwlClass] = [root]
        seen: set[tuple[type[PydOwlClass], str]] = set()

        while stack:
            inst = stack.pop()
            key = (type(inst), inst.identifier)
            if key in seen:
                continue
            seen.add(key)
            cls.register(inst)

            inst_type = type(inst)
            for field_name, field_info in inst_type.model_fields.items():
                if field_name in ("identifier", "pydowl_version"):
                    continue
                value = getattr(inst, field_name)
                if value is None:
                    continue

                category = identify_field_type_category(field_info, field_name)
                if category == FieldTypeCategory.OPTIONAL_PYD_CLS:
                    assert isinstance(value, PydOwlClass)
                    stack.append(value)
                elif category == FieldTypeCategory.LIST_PYD_CLS:
                    assert isinstance(value, list)
                    for item in value:
                        if isinstance(item, PydOwlClass):
                            stack.append(item)

    @classmethod
    def get(cls, model_type: type[T], unique_id: str) -> Optional[T]:
        """
        Retrieve a previously-registered instance if it is still alive.
        """
        with cls._lock:
            ref = cls._registry.get(model_type, {}).get(unique_id)
            if ref:
                obj = ref()
                if obj is not None and isinstance(obj, model_type):
                    return cast(T, obj)
            return None

    @classmethod
    def clear(cls) -> None:
        """
        Clear the entire registry.

        Intended mainly for tests or batch jobs that want to reset
        in-memory identity between runs.
        """
        with cls._lock:
            cls._registry.clear()

    @classmethod
    def delete(cls, model_type: type[T], unique_id: str) -> None:
        """
        Remove a specific entry from the registry, if present.
        """
        with cls._lock:
            if unique_id in cls._registry.get(model_type, {}):
                del cls._registry[model_type][unique_id]
