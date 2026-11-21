from groq import Groq
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM


class GroqJudgeAdapter(DeepEvalBaseLLM):
    """Adapts Groq SDK to DeepEval's DeepEvalBaseLLM interface."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self._model_name = model_name
        self._client = Groq(api_key=api_key)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: BaseModel = None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self._model_name, "messages": messages, "temperature": 0.0}

        if schema:
            kwargs["response_format"] = {"type": "json_object"}
            messages[0]["content"] += f"\n\nReturn strictly valid JSON matching: {schema.model_json_schema()}"

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return schema.model_validate_json(content) if schema else content

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"groq/{self._model_name}"