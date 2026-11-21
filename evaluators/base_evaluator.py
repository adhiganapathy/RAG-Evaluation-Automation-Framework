from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvalVerdict:
    passed: bool
    metric_name: str
    score: float
    reason: str = ""


class BaseEvaluator(ABC):
    """Abstract Strategy interface for all AI quality and security evaluators."""

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> list[EvalVerdict]:
        pass