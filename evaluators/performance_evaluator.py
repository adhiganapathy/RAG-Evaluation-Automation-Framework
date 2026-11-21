from evaluators.base_evaluator import BaseEvaluator, EvalVerdict
from pages.rag_page import RAGResultDTO


class PerformanceEvaluator(BaseEvaluator):
    """Evaluates end-to-end RAG response latency and verbosity/token budget."""

    def __init__(self, max_latency_sec: float = 15.0, max_response_words: int = 150):
        self._max_latency = max_latency_sec
        self._max_words = max_response_words

    def evaluate(self, rag_result: RAGResultDTO) -> list[EvalVerdict]:
        latency_passed = rag_result.latency_sec <= self._max_latency
        word_count = len(rag_result.answer.split())
        budget_passed = word_count <= self._max_words

        return [
            EvalVerdict(
                passed=latency_passed,
                metric_name="ResponseLatency",
                score=rag_result.latency_sec,
                reason=f"Took {rag_result.latency_sec}s (SLA limit: {self._max_latency}s)",
            ),
            EvalVerdict(
                passed=budget_passed,
                metric_name="TokenBudgetWords",
                score=float(word_count),
                reason=f"Generated {word_count} words (max budget: {self._max_words} words)",
            ),
        ]