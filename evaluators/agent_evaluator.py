import json
from groq import Groq
from evaluators.base_evaluator import BaseEvaluator, EvalVerdict
from pages.agent_page import AgentTurnDTO


class AgentEvaluator(BaseEvaluator):
    """Evaluates Multi-Turn Agentic Memory and Goal Completion across full chat history."""

    def __init__(self, groq_api_key: str, judge_model: str, min_score: float = 0.75):
        self._groq = Groq(api_key=groq_api_key)
        self._model = judge_model
        self._min_score = min_score

    def evaluate(
        self,
        agent_result: AgentTurnDTO,
        goal: str,
        transcript_history: list[str] | None = None,
    ) -> list[EvalVerdict]:
        if transcript_history:
            transcript = "\n".join(transcript_history)
        else:
            transcript = f"FINAL ASSISTANT REPLY: {agent_result.latest_reply}"

        prompt = (
            "You are an AI Agent QA Evaluator. Grade this multi-turn conversation from 0.0 to 1.0 "
            "based on: (1) Context retention across turns, and (2) Completing the user's goal accurately.\n"
            f"Expected Goal: {goal}\n"
            f"Conversation Transcript:\n{transcript}\n"
            'Reply strictly in JSON: {"score": 0.0_to_1.0, "reason": "brief explanation"}'
        )

        res = self._groq.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(res.choices[0].message.content)
        score = float(data.get("score", 0.0))

        return [
            EvalVerdict(
                passed=score >= self._min_score,
                metric_name="ConversationCompleteness",
                score=round(score, 3),
                reason=data.get("reason", ""),
            )
        ]