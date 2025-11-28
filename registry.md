###  Registry

`PydOwlRegistry` is an optional, **process-local identity map**:

```python
{ (PydOwlClass subclass, identifier) -> weakref(instance) }
```

### How it’s used implicitly

The registry is populated whenever you go through `from_data`:

* `PydOwlClass.from_data(data)`
  (called by many internal paths)
* `PydOwlClass.from_mongo_docs(...)`
* `PydOwlClass.pull_owlready(...)`
* `pydowl.sparql.pull` (both full and ABox modes)

Those helpers all:

1. Look up `(cls, identifier)` in the registry.
2. If an instance is present and still alive:

    * call `instance.update(**data)` and return it.
3. Otherwise:

    * create a new instance via `model_validate`,
    * register it,
    * return it.

This gives you:

* **Cross-call identity reuse**
  Loading the same `(cls, id)` multiple times in a process yields the same Python object, updated in-place.

* **Merging overlapping subgraphs**
  If two separate pulls touch the same node, they converge to the same instance rather than diverging into duplicates.

### When to clear it

It’s usually **not** necessary to interact with the registry in simple scripts. It’s helpful when:

* You’re writing tests and want strict isolation between tests:

  ```python
  from pydowl import PydOwlRegistry

  PydOwlRegistry.clear()
  # run test
  PydOwlRegistry.clear()
  ```

* You’re in a long-lived process and want to explicitly “reset” identity between batches or requests.

### When to register manually

Manual registration is only needed when you:

* Construct instances **by hand** and want later loads to merge into those exact instances:

  ```python
  from pydowl import PydOwlRegistry

  alice = Person(identifier="alice", name="Alice")
  PydOwlRegistry.register(alice)

  # later: from SPARQL/Mongo
  alice2 = Person.from_mongo_docs("alice", docs)
  assert alice2 is alice  # same instance, updated
  ```

* Want to seed the registry with an entire graph:

  ```python
  from pydowl import PydOwlRegistry

  root = build_my_graph()
  PydOwlRegistry.register_graph(root)
  ```

  This walks `OPTIONAL_PYD_CLS` and `LIST_PYD_CLS` fields, registering every reachable node. Later pulls (from Mongo or SPARQL/Owlready) will reuse those instances instead of creating new ones.

If you don’t care about strict in-process identity reuse, you can happily ignore `PydOwlRegistry` and let the default behaviour handle it.

