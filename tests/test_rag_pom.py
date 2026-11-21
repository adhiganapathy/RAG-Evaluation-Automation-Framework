import pytest
from framework_core.data_loader import DataLoader
from pages.rag_page import RAGPage
from evaluators.rag_evaluator import RAGEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from evaluators.safety_evaluator import SafetyEvaluator

RAG_DATASET = DataLoader.load_json("rag_cases.json")


class TestRAGDataDriven:
    @pytest.mark.parametrize("case", RAG_DATASET, ids=[c["test_id"] for c in RAG_DATASET])
    def test_rag_quality_and_grounding(
        self,
        request,
        rag_page: RAGPage,
        rag_evaluator: RAGEvaluator,
        performance_evaluator: PerformanceEvaluator,
        safety_evaluator: SafetyEvaluator,
        case: dict,
    ):
        result = rag_page.execute(case["query"])
        verdicts = performance_evaluator.evaluate(result)

        if case["evaluate_faithfulness"]:
            verdicts.extend(
                rag_evaluator.evaluate(
                    result,
                    expected_output=case.get("expected_output", ""),
                    is_out_of_scope=case.get("is_out_of_scope", False),
                )
            )
            verdicts.extend(safety_evaluator.evaluate(result))

        request.node.eval_summary = " | ".join(
            f"{v.metric_name}: {v.score:.2f} ({'PASS' if v.passed else 'FAIL'})" for v in verdicts
        )

        for kw in case["expected_keywords"]:
            assert kw.lower() in result.answer.lower(), (
                f"[{case['test_id']}] Missing expected keyword '{kw}' in answer: {result.answer}"
            )

        if case["require_citations"]:
            assert rag_evaluator.has_valid_inline_citations(result.answer), (
                f"[{case['test_id']}] Expected inline citations, got: {result.answer}"
            )

        for verdict in verdicts:
            assert verdict.passed, (
                f"[{case['test_id']} - {verdict.metric_name}] Score {verdict.score}: {verdict.reason}"
            )