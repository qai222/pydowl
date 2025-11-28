# AGENTS.md

Guidelines for automated coding agents (and humans using them) working on **pydowl**.

## 1. Scope

Agents **may**:

- Edit code under `pydowl/` (core models, datatypes, SPARQL helpers, etc.).
- Add/modify tests that run **fully offline**.
- Run local unit tests that do **not** require network access or cloud services.

Agents **must NOT**:

- Make real network calls to SPARQL endpoints.
- Make real network calls to Azure (Blob Storage or otherwise).
- Depend on private credentials or secrets.

Live SPARQL / Azure integration is verified manually by a human.

---

## 2. Tests

When running tests, agents should restrict to offline tests only.

Recommended command:

```bash
pytest tests -k "not sparql and not azure"
````

In particular, **do not run**:

* `tests/test_sparql_large_node.py`
* `tests/test_azure.py`
* or any other test that:

    * requires a real SPARQL endpoint, or
    * requires real Azure credentials / storage.

If you add new networked tests, mark them so they are skipped by default (e.g. via `pytest.mark.skip` or a skip
condition on an env var), and document that they are **manual-only**.

---

## 3. Implementation notes

* Prefer preserving existing public APIs and validation behaviour unless there is a clear reason to change them.
* If you change SPARQL/Azure-related code paths, add or update **offline** tests that use mocking/monkeypatching instead
  of real network calls, and mention in your notes that live tests should be re-run manually.

