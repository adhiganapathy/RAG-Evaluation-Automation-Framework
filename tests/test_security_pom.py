import pytest
from framework_core.data_loader import DataLoader
from pages.rag_page import RAGPage
from evaluators.security_evaluator import SecurityEvaluator

SECURITY_DATASET = DataLoader.load_json("security_cases.json")


class TestSecurityDataDriven:
    @pytest.mark.parametrize("case", SECURITY_DATASET, ids=[c["test_id"] for c in SECURITY_DATASET])
    def test_redteam_guardrails(self, request, rag_page: RAGPage, security_evaluator: SecurityEvaluator, case: dict):
        result = rag_page.execute(case["prompt"])
        verdicts = security_evaluator.evaluate(result)
        request.node.eval_summary = " | ".join(
            f"{v.metric_name}: {'PASS' if v.passed else 'BREACH'} ({v.reason})" for v in verdicts
        )

        for verdict in verdicts:
            assert verdict.passed, (
                f"[{case['test_id']} | {case['attack_category']}] Failed {verdict.metric_name}: {verdict.reason}"
            )