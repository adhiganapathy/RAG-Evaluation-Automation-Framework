from deepeval.metrics import ToxicityMetric, BiasMetric
from deepeval.test_case import LLMTestCase
from config.settings import EvalThresholds
from evaluators.base_evaluator import BaseEvaluator, EvalVerdict
from framework_core.llm_judge import GroqJudgeAdapter
from pages.rag_page import RAGResultDTO


class SafetyEvaluator(BaseEvaluator):
    """Evaluates AI responses for Toxic/Abusive tone and Demographic/Opinion Bias."""

    def __init__(self, judge_llm: GroqJudgeAdapter, thresholds: EvalThresholds):
        self._judge = judge_llm
        self._thresholds = thresholds

    def evaluate(self, rag_result: RAGResultDTO) -> list[EvalVerdict]:
        test_case = LLMTestCase(
            input=rag_result.query,
            actual_output=rag_result.answer,
        )

        metrics = [
            (
                "ToxicityCheck",
                ToxicityMetric(
                    threshold=self._thresholds.toxicity_max,
                    model=self._judge,
                    include_reason=True,
                ),
            ),
            (
                "BiasCheck",
                BiasMetric(
                    threshold=self._thresholds.bias_max,
                    model=self._judge,
                    include_reason=True,
                ),
            ),
        ]

        verdicts: list[EvalVerdict] = []
        for name, metric in metrics:
            metric.measure(test_case)
            verdicts.append(
                EvalVerdict(
                    passed=metric.is_successful(),
                    metric_name=name,
                    score=round(metric.score or 0.0, 3),
                    reason=metric.reason or "",
                )
            )
        return verdicts