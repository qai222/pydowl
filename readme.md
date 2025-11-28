# pydowl

**pydowl** bridges **Pydantic** models and **OWL / Graph DBs**.

Define your ontology schema as Pydantic models, and pydowl handles:
- projecting instances into **Owlready2**,
- pushing/pulling to **SPARQL 1.1 endpoints** (Virtuoso, GraphDB, …),
- storing node-centric views in **MongoDB**,
- encoding complex Python types as custom OWL datatypes,
- and optionally offloading large literals to **Azure Blob Storage**.

![pydowl](docs/float/pydowl.png)

---

## Installation

```bash
pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/Abstrax-Tech/pydowl
````

---

## Quickstart
### 1. Define models
```python
import owlready2
from datetime import datetime
from typing import Optional, List
from pydantic import Field
from pydowl import PydOwlClass
"""
* `PydOwlClass` is a Pydantic `BaseModel` with OWL / Mongo / SPARQL glue. Each instance has an `identifier` used as:
  * the IRI fragment in Owlready2 and SPARQL,
  * the `_id` in Mongo node-centric docs.
* A discriminator `pydowl_type="<module>:<QualName>"` is injected on serialisation and used to up-cast on deserialisation.
"""

class Address(PydOwlClass):
    street: Optional[str] = None
    city: Optional[str] = None

class Person(PydOwlClass):
    name: Optional[str] = None
    age: Optional[int] = None
    registered: Optional[datetime] = None
    addresses: List[Address] = Field(default_factory=list)
```

---

### 2. Push into Owlready2
```python
onto = owlready2.get_ontology("http://example.org/my.owl#")

home = Address(identifier="addr1", street="123 Main St", city="Springfield")
office = Address(identifier="addr2", street="1 Infinite Loop", city="Cupertino")

alice = Person(
    identifier="alice",
    name="Alice",
    age=30,
    registered=datetime.now(),
    addresses=[home, office],
)

# Create classes and properties dynamically based on the Pydantic schema
alice.push_owlready(onto, dynamic_tbox=True)

# Now you can explore the ontology using Owlready2:
addr1 = onto.search_one(iri="http://example.org/my.owl#addr1")
residents = addr1.INVERSE_has_addresses  # Owlready2 inverse
print([p.has_name for p in residents])   # ['Alice']
print([a.has_city for p in residents for a in p.has_addresses])
# ['Springfield', 'Cupertino']

# And you can pull individuals back into Pydantic models
alice_copy = Person.pull_owlready(
    onto,
    onto.search_one(iri=onto.base_iri + "alice"),
)
assert isinstance(alice_copy, Person)
assert alice_copy.addresses[0].city == "Springfield"
```

---

### 3. Push to a SPARQL endpoint

```python
from pydowl.sparql import SparqlEndpoint, push_owlready_to_sparql_endpoint

endpoint = SparqlEndpoint(
    url="https://sparql.example.org/sparql-auth/",
    username="user",
    password="pass",
)

# Push TBox + ABox into the named graph = onto.base_iri
push_owlready_to_sparql_endpoint(onto, endpoint, abox_only=False)

# You can also push a single `PydOwlClass` instance via the helper
from pydowl.sparql import push_pyd_to_sparql_endpoint
push_pyd_to_sparql_endpoint(alice, endpoint, abox_only=False)
#  If you pass `tbox_ontology_path=...`, pydowl will load that TBox and disable dynamic TBox creation; 
# otherwise, a temporary ontology with dynamic TBox is used.

```

---

### 4. Pull from SPARQL (full graph vs ABox-only)

#### 


```python
from pydowl.sparql import pull_pyd_from_sparql_endpoint

"""
Full-graph mode (Tier 1)
Load the entire named graph into Owlready2 and delegate to `Person.pull_owlready`:
Use this when:
* the named graph is small/medium,
* you want a complete ontology in memory (for reasoning, debugging),
* and the SPARQL TBox is broadly consistent with your Pydantic schema.
"""

alice_full = pull_pyd_from_sparql_endpoint(
    ind_iri=onto.base_iri + "alice",
    graph_iri=onto.base_iri,
    sparql_ep=endpoint,
    mode="full",  # default, can be omitted
)

assert isinstance(alice_full, Person)
print(alice_full.addresses[0].city)  # 'Springfield'

"""
ABox-only mode (Tier 2)
Treat the SPARQL store as a fact store and let the schema drive what to fetch:
In `"abox"` mode:
* For each field in `Person`, pydowl knows the property IRI is `<graph_iri>has_<field_name>`.
* It issues small `SELECT` queries for:
  * scalar fields (`OPTIONAL_PY_LITERAL`, `OPTIONAL_DATETIME`, enums, JSON blobs, NumPy arrays),
  * nested `PydOwlClass` / `list[PydOwlClass]` fields (`OPTIONAL_PYD_CLS`, `LIST_PYD_CLS`) up to `max_depth`.
* It decodes literals using the builtin and custom datatype tables and materialises a `Person` via `from_data`.
This mode is ideal for application code that wants to hydrate a few individuals without loading an entire graph.
"""
alice_abox = pull_pyd_from_sparql_endpoint(
    ind_iri=onto.base_iri + "alice",
    graph_iri=onto.base_iri,
    sparql_ep=endpoint,
    mode="abox",
    max_depth=2,  # recursion depth into nested PydOwlClass fields
)

assert isinstance(alice_abox, Person)
print([a.city for a in alice_abox.addresses])
```

---

### 5. MongoDB integration
```python
"""
You can use MongoDB as your canonical data store, with OWL/SPARQL as projections.
"""

# Create object graph
alice = Person(...)

# Serialise to node-centric Mongo docs
docs = alice.to_mongo_docs()
# Example doc for Person:
# {
#   "_id": "alice",
#   "pydowl_type": "my_module:Person",
#   "name": "Alice",
#   "addresses": ["addr1", "addr2"],
#   ...
# }

from pymongo import MongoClient
client = MongoClient(...)
db = client["mydb"]

collection = db[Person.mongo_collection_name()]  # "persons" by default
collection.insert_many(docs)

# Later, you can hydrate a graph from the stored docs: 
docs = list(collection.find({"_id": {"$in": ["alice", "addr1", "addr2"]}}))
# `from_mongo_docs` rebuilds the graph from IDs, and will reuse instances for the same `(cls, identifier)` 
# within a process when they’re present in the registry.
alice2 = Person.from_mongo_docs("alice", docs)

assert isinstance(alice2, Person)
assert alice2.identifier == "alice"
assert alice2.addresses[0].street == "123 Main St"

```


---

### 6. Custom datatypes 
```python
import pydantic_numpy.typing as pnd
from pydantic import Field
from typing import Dict, Any, Optional
"""
pydowl defines custom OWL datatypes under `http://pydowl.org/dtype#`:
* `JsonStringDatatype` – arbitrary JSON serialisable values.
* `NpArrayNdDatatype` – n-D NumPy arrays (except 2D).
* `NpArray2dDatatype` – 2D arrays (stored as Parquet+Snappy).
* `DateTimeDatatype` – timezone-aware datetimes.
You get them by using specific field types:
"""

class BlobHolder(PydOwlClass):
    # Treated as JSON blob (PY_JSON → JsonStringDatatype)
    meta: Dict[str, Any] = Field(default_factory=dict)

    # 2D NumPy array (OPTIONAL_NP2D_ARRAY → NpArray2dDatatype)
    arr: Optional[pnd.Np2DArray] = None
```

### 7. Large literals offload


```python
from datetime import datetime
import numpy as np
import owlready2
from typing import Optional, Dict, Any
import pydantic_numpy.typing as pnd
from pydantic import Field
from pydowl import PydOwlClass
from pydowl.sparql import (
    SparqlEndpoint,
    push_owlready_to_sparql_endpoint,
    pull_pyd_from_sparql_endpoint,
)
"""
To offload large literals to Azure Blob Storage, configure:

export AZURE_STORAGE_CONNECTION_STRING="..."
export AZURE_LARGE_NODE_CONTAINER_SAS_TOKEN="..."

Then any large data property (e.g. a huge JSON blob or big NumPy array) can be offloaded during SPARQL push.
"""

# A model with "large" fields
class LargeNode(PydOwlClass):
    meta: Dict[str, Any] = Field(default_factory=dict)     # JSON blob
    arr: Optional[pnd.Np2DArray] = None                    # 2D NumPy array
    ts: Optional[datetime] = None                          # datetime

# Build an ontology & instance
onto = owlready2.get_ontology("http://example.org/large.owl#")

huge_json = {
    "text": "A" * 200_000,
    "nums": list(range(50_000)),
}
huge_array = np.arange(10_000, dtype=np.int32).reshape((100, 100))

node = LargeNode(
    identifier="ln1",
    meta=huge_json,
    arr=huge_array,
    ts=datetime.now(),
)

node.push_owlready(onto, dynamic_tbox=True)

# Push to SPARQL; large literals will be offloaded automatically
endpoint = SparqlEndpoint(
    url="https://sparql.example.org/sparql-auth/",
    username="user",
    password="pass",
)

push_owlready_to_sparql_endpoint(onto, endpoint, abox_only=False)
"""
This push would
* detect large data literals (by UTF-8 size),
* store a JSON record `{"datatype": dt_iri, "value": value}` in Azure Blob Storage,
* replace the literal with a URL literal ending with `__LARGE_NODE`.
"""

# Later: pull back from SPARQL and verify content
node_copy = pull_pyd_from_sparql_endpoint(
    ind_iri=onto.base_iri + "ln1",
    graph_iri=onto.base_iri,
    sparql_ep=endpoint,
    mode="full",    # or "abox"
)
"""
* URL placeholders are detected,
* The record is downloaded,
* The original value is decoded into:
  * the appropriate custom datatype wrapper in Owlready (full mode), or
  * bare Python types in ABox mode.
"""

assert isinstance(node_copy, LargeNode)
assert node_copy.meta == huge_json
assert np.array_equal(node_copy.arr, huge_array)
```

---

### 8. Value objects with `PydOwlDataClass`


```python
from typing import Optional
from pydowl import PydOwlDataClass
"""
`PydOwlDataClass` is a value-object variant of `PydOwlClass` with stable semantic IDs and strict immutability for identity fields.
"""

class Quantity(PydOwlDataClass):
    __id_fields__ = ("value", "unit")  
    # identity is (value, unit) only. 
    # if not set explicitly this includes all fields except "identifier" and "pydowl_version"

    value: Optional[float] = None
    unit: Optional[str] = None
    note: Optional[str] = None  # non-identity

q = Quantity(value=1.0, unit="kg")
# if `identifier` is not provided, a stable ID is computed as "<module>:<QualName>:<16-hex-sha256>"

q.note = "new"           # OK (non-identity)
q.update(note="newer")   # OK

# Identity fields are immutable after initialisation:
q.value = 2.0            # raises TypeError
q.update(value=2.0)      # raises TypeError
"""
`from_data` / `update` enforce the same rule: 
any merge that would change identity fields raises `TypeError`. 
If you need a different logical value, you create a new instance.
This gives you true value semantics for nodes like quantities, configurations, etc., 
and makes deduplication by ID reliable.
"""
```

---



## Wishlist (high-level)

A few directions to explore next:

* **Automatic inverse properties** in the generated TBox (for navigation / reasoning).
* **Pluggable large-literal backends** (S3, GCS, local fs) behind a simple storage interface.
* **SHACL validation** on pull (using pySHACL) before hydrating into Pydantic.
* **Reasoner integration** (HermiT/ELK) for classification and consistency checking in complex ontologies.
