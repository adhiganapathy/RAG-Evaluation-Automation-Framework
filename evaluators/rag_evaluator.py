import re
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from evaluators.base_evaluator import BaseEvaluator, EvalVerdict
from pages.rag_page import RAGResultDTO
from config.settings import EvalThresholds


class RAGEvaluator(BaseEvaluator):
    """Evaluates Retrieval Similarity, Faithfulness, Relevancy, Hallucination, Precision, and Recall."""

    def __init__(self, judge_llm, thresholds: EvalThresholds):
        self._judge = judge_llm
        self._thresholds = thresholds

    def evaluate(
        self,
        rag_result: RAGResultDTO,
        expected_output: str = "",
        is_out_of_scope: bool = False,
    ) -> list[EvalVerdict]:
        verdicts = []

        max_sim = max(rag_result.similarity_scores, default=0.0)
        verdicts.append(EvalVerdict(
            passed=max_sim >= self._thresholds.retrieval_similarity,
            metric_name="RetrievalSimilarity",
            score=round(max_sim, 3),
            reason=f"Top chunk score {max_sim:.3f} vs threshold {self._thresholds.retrieval_similarity}",
        ))

        test_case = LLMTestCase(
            input=rag_result.query,
            actual_output=rag_result.answer,
            expected_output=expected_output or None,
            retrieval_context=rag_result.contexts,
            context=rag_result.contexts,
        )

        metrics = [
            ("Faithfulness", FaithfulnessMetric(threshold=self._thresholds.faithfulness, model=self._judge)),
            ("AnswerRelevancy", AnswerRelevancyMetric(threshold=self._thresholds.answer_relevancy, model=self._judge)),
        ]

        if expected_output:
            metrics.append(
                ("Hallucination", HallucinationMetric(threshold=self._thresholds.hallucination_max, model=self._judge))
            )
            if not is_out_of_scope:
                metrics.extend([
                    ("ContextualPrecision", ContextualPrecisionMetric(threshold=self._thresholds.contextual_precision, model=self._judge)),
                    ("ContextualRecall", ContextualRecallMetric(threshold=self._thresholds.contextual_recall, model=self._judge)),
                ])

        for name, metric in metrics:
            metric.measure(test_case)
            verdicts.append(EvalVerdict(
                passed=metric.is_successful(),
                metric_name=name,
                score=round(metric.score or 0.0, 3),
                reason=metric.reason or "",
            ))

        return verdicts

    def has_valid_inline_citations(self, answer: str) -> bool:
        return len(re.findall(r"\[\d+\]", answer)) > 0