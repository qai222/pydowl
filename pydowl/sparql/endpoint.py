from typing import Literal, Any, Dict, cast

from SPARQLWrapper import SPARQLWrapper, JSON, DIGEST
from pydantic import BaseModel

Method = Literal["GET", "POST"]
ReturnFormat = str  # SPARQLWrapper uses string constants (e.g. JSON)


class SparqlEndpoint(BaseModel):
    """
    Thin wrapper around :class:`SPARQLWrapper.SPARQLWrapper`.

    Use:

    * :meth:`query_select` for SELECT queries (returns list of bindings).
    * :meth:`update` for INSERT / DELETE / CLEAR / etc. (no return value).
    * :meth:`query` as a convenience that dispatches based on the query
      string (SELECT vs other).
    """

    url: str
    username: str
    password: str
    digest_auth: bool = True

    def setup_wrapper(
        self,
        method: Method = "POST",
        return_format: ReturnFormat = JSON,
    ) -> SPARQLWrapper:
        """
        Create and configure a SPARQLWrapper instance for this endpoint.
        """
        sparql = SPARQLWrapper(self.url)
        sparql.setMethod(method)
        sparql.setReturnFormat(return_format)
        sparql.setCredentials(self.username, self.password)
        if self.digest_auth:
            sparql.http_auth = DIGEST
        return sparql

    def query_select(
        self,
        q: str,
        method: Method = "POST",
    ) -> list[Dict[str, Any]]:
        """
        Execute a SPARQL SELECT and return ``results['bindings']``.
        """
        sparql = self.setup_wrapper(method=method, return_format=JSON)
        sparql.setQuery(q)
        results = sparql.query().convert()
        data = cast(Dict[str, Any], results)
        return cast(list[Dict[str, Any]], data.get("results", {}).get("bindings", []))

    def update(
        self,
        q: str,
        method: Method = "POST",
    ) -> None:
        """
        Execute a SPARQL UPDATE (INSERT/DELETE/CLEAR/etc.).

        No result is returned; any failure should surface as an
        exception from SPARQLWrapper.
        """
        # We still use JSON as return format, but ignore the payload.
        sparql = self.setup_wrapper(method=method, return_format=JSON)
        sparql.setQuery(q)
        # For Virtuoso / GraphDB this will perform the update; we
        # deliberately do not attempt to inspect the result.
        sparql.query()

    def query(
        self,
        q: str,
        method: Method = "POST",
        return_format: ReturnFormat = JSON,
    ) -> Any:
        """
        Convenience method:

        * If the query starts with ``SELECT`` (case-insensitive), dispatch
          to :meth:`query_select` and return bindings (JSON result format
          is always used internally, regardless of ``return_format``).
        * Otherwise, treat it as an UPDATE and call :meth:`update`;
          returns ``None``.

        Existing tests and fixtures (e.g. using ``CLEAR GRAPH``) can
        continue to call ``query(...)`` without worrying about return
        types.
        """
        stripped = q.lstrip().upper()
        if stripped.startswith("SELECT"):
            return self.query_select(q, method=method)
        # UPDATE / CLEAR / INSERT DATA / etc.
        self.update(q, method=method)
        return None
