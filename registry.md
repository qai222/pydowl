## Registry

`PydOwlRegistry` is a **process-local, weakly referenced identity map** for `PydOwlClass` instances. It keeps at most one live Python object per `(concrete_model_class, identifier)` pair while that object is still strongly referenced somewhere else in your process.

### How it is populated automatically

`from_data` registers every instance it creates, and all higher-level loaders delegate to it:

- `PydOwlClass.from_mongo_docs(...)`
- `PydOwlClass.pull_owlready(...)`
- SPARQL pulls in both full-graph and ABox modes

The sequence is consistent:

1. Look up `(cls, identifier)` in the registry.
2. If found and still alive, **reuse** the instance and call `update(**data)` on it.
3. Otherwise, create, register, and return a new instance.

### Why it exists

- **Identity reuse across calls:** repeated pulls of the same node share one object instead of silent duplicates.
- **Merging overlapping subgraphs:** separate loads that touch the same node converge on the same instance, so updates merge instead of drifting apart.
- **Thread-safe and disposable:** lookups/updates are guarded by a lock; entries are weak refs so garbage collection can reclaim unused objects without manual cleanup.

### When to clear it

The registry is process-local only; it is not persisted between runs. Clear it when you want hard isolation:

```python
from pydowl import PydOwlRegistry

PydOwlRegistry.clear()  # e.g., between tests or batch jobs
```

You can also `delete(cls, identifier)` to remove one entry. Clearing is especially helpful in test suites where identity reuse between cases would mask bugs.

### When to register manually

You rarely need to touch the registry directly. Consider it when:

- **Manually constructed graphs should win:** you build objects from an API payload and want subsequent Mongo/OWL/SPARQL pulls to merge into the same instances.
- **Seeding identity before hydration:** call `PydOwlRegistry.register_graph(root)` to register every reachable `OPTIONAL_PYD_CLS` / `LIST_PYD_CLS` node before a pull, ensuring overlap collapses onto your pre-built objects.

Minimal example:

```python
from pydowl import PydOwlRegistry

alice = Person(identifier="alice", name="Alice")
PydOwlRegistry.register(alice)

# Later: hydrate from storage
alice2 = Person.from_mongo_docs("alice", docs)
assert alice2 is alice  # same instance, updated via `update`
```

### Practical limits

- **Weak references only:** if no other code holds a strong reference, the instance may be garbage collected and recreated on the next pull.
- **No cross-process guarantees:** each process has its own registry; you still need canonical storage (SPARQL, Mongo, etc.) for durability.

Keeping these constraints in mind lets you use the registry as a simple, fail-fast cache of in-memory identity rather than as a persistence layer.
