from typing import Optional, List

import pytest
from pydantic import Field
import enum
from pydowl import PydOwlClass


class Country(enum.StrEnum):
    USA = "USA"
    UK = "UK"
    CHN = "CHN"


class Plant(PydOwlClass):
    country_origin: Optional[Country] = Field(default=None)

    age: Optional[float] = Field(default=None)

    cultivators: List[str] = Field(default_factory=list)


class Component(PydOwlClass):
    smiles: Optional[str] = Field(default=None)

    amount: Optional[float] = Field(default=None)

    amount_unit: Optional[str] = Field(default=None)


class Tree(Plant):
    expected_components: List[Component] = Field(default_factory=list)

    related_plants: List[Plant] = Field(default_factory=list)


class PlantExtract(PydOwlClass):
    plant_origin: Optional[Plant] = Field(default=None)

    components: List[Component] = Field(default_factory=list)


@pytest.fixture(scope="function", autouse=False)
def plant_schema_entities():
    plant1 = Plant(
        identifier="plant1",
        country_origin="USA",
        age=42,
        cultivators=["farmer1", "farmer2"],
    )
    plant2 = Plant(
        identifier="plant2",
        country_origin="USA",
        age=49,
        cultivators=["farmer1", "farmer3"],
    )
    component1 = Component(
        identifier="component1", smiles="CCO", amount=25, amount_unit="%"
    )
    component2 = Component(
        identifier="component2", smiles="OCCCCO", amount=25, amount_unit="g"
    )
    tree1 = Tree(
        identifier="tree1",
        country_origin="UK",
        age=12,
        cultivators=["farmer1", "farmer4"],
        related_plants=[plant1, plant2],
        expected_components=[component1, component2],
    )
    tree2 = Tree(
        identifier="tree2",
        country_origin="UK",
        age=19,
        cultivators=["farmer1", "farmer2"],
        related_plants=[plant1],
        # related_plants=[tree1],
        expected_components=[component1],
    )
    # tree1.related_plants.append(tree2)  # this would cause recursion errors
    tree1_extract = PlantExtract(
        identifier="tree1_extract",
        plant_origin=tree1,
        components=[component1, component2],
    )
    tree2_extract = PlantExtract(
        identifier="tree2_extract",
        plant_origin=tree2,
        components=[component2],
    )
    return {
        "plant1": plant1,
        "plant2": plant2,
        "component1": component1,
        "component2": component2,
        "tree1": tree1,
        "tree2": tree2,
        "tree1_extract": tree1_extract,
        "tree2_extract": tree2_extract,
    }
