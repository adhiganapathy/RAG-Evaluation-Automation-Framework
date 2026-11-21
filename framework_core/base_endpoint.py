from abc import ABC, abstractmethod
from r2r import R2RClient


class BaseEndpoint(ABC):
    """Abstract Base Page Object representing an R2R API capability."""

    def __init__(self, client: R2RClient):
        self._client = client

    @property
    def default_search_settings(self) -> dict:
        return {
            "use_hybrid_search": False,
            "graph_search_settings": {"enabled": False},
        }

    def health_check(self) -> bool:
        res = self._client.system.health()
        return res.get("results", {}).get("message") == "ok"

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass