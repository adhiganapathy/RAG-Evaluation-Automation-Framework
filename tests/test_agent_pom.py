import pytest
from framework_core.data_loader import DataLoader
from pages.agent_page import AgentPage
from evaluators.agent_evaluator import AgentEvaluator

AGENT_DATASET = DataLoader.load_json("agent_conversations.json")


class TestAgentDataDriven:
    @pytest.mark.parametrize("scenario", AGENT_DATASET, ids=[s["test_id"] for s in AGENT_DATASET])
    def test_multiturn_conversations(
        self, request, agent_page: AgentPage, agent_evaluator: AgentEvaluator, scenario: dict
    ):
        turn_summaries = []
        full_transcript = []

        for step_idx, turn in enumerate(scenario["turns"], 1):
            response = agent_page.execute(turn["user_message"])
            full_transcript.append(f"Turn {step_idx} USER: {turn['user_message']}")
            full_transcript.append(f"Turn {step_idx} ASSISTANT: {response.latest_reply}")
            turn_summaries.append(f"Turn {step_idx}: Verified {turn['expected_contains']}")

            for expected in turn["expected_contains"]:
                assert expected.lower() in response.latest_reply.lower(), (
                    f"[{scenario['test_id']} Turn {step_idx}] Expected '{expected}', got: {response.latest_reply}"
                )

        verdicts = agent_evaluator.evaluate(
            response, goal=scenario["scenario"], transcript_history=full_transcript
        )
        agent_scores = " | ".join(
            f"{v.metric_name}: {v.score:.2f} ({'PASS' if v.passed else 'FAIL'})" for v in verdicts
        )
        request.node.eval_summary = f"{' | '.join(turn_summaries)} | {agent_scores}"

        for verdict in verdicts:
            assert verdict.passed, f"[{scenario['test_id']} - {verdict.metric_name}] {verdict.reason}"