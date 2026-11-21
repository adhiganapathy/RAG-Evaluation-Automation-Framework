from dataclasses import dataclass, field
from framework_core.base_endpoint import BaseEndpoint



@dataclass
class AgentTurnDTO:
    conversation_id: str
    latest_reply: str
    tool_calls: list[dict] = field(default_factory=list)
    full_history: list[dict] = field(default_factory=list)


class AgentPage(BaseEndpoint):
    """Page Object for R2R Stateful Multi-Turn & Tool-Calling Agent."""

    def __init__(self, client):
        super().__init__(client)
        self._conversation_id: str | None = None

    def reset_session(self) -> None:
        self._conversation_id = None

    def execute(self, user_message: str) -> AgentTurnDTO:
        payload = {
            "message": {"role": "user", "content": user_message},
            "search_settings": self.default_search_settings,
        }
        if self._conversation_id:
            payload["conversation_id"] = self._conversation_id

        raw_res = self._client.retrieval.agent(**payload)
        return self._parse_agent_response(raw_res)

    def _parse_agent_response(self, raw_res: dict) -> AgentTurnDTO:
        results = raw_res.get("results", {})
        self._conversation_id = results.get("conversation_id", self._conversation_id)
        messages = results.get("messages", [])

        assistant_replies = [
            m.get("content", "") for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]
        tool_calls = [
            tc for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]

        return AgentTurnDTO(
            conversation_id=self._conversation_id or "",
            latest_reply=assistant_replies[-1] if assistant_replies else "",
            tool_calls=tool_calls,
            full_history=messages,
        )