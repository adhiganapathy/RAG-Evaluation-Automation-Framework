import json
from pathlib import Path
from groq import Groq


class SyntheticTestGenerator:
    """Generates golden RAG Q&A test cases dynamically from a knowledge base document."""

    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self._client = Groq(api_key=groq_api_key)
        self._model = model_name

    def generate_from_document(self, doc_path: str, count: int = 3) -> list[dict]:
        content = Path(doc_path).read_text(encoding="utf-8")
        prompt = (
            f"Read this policy document and generate {count} customer support test cases.\n"
            "Do NOT include confidential internal escalation contacts.\n"
            f"Document:\n{content}\n\n"
            'Return strictly valid JSON: {"cases": [{"test_id": "SYN_001", "description": "...", '
            '"query": "...", "expected_keywords": ["..."], "require_citations": true, "evaluate_faithfulness": true}]}'
        )
        res = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(res.choices[0].message.content).get("cases", [])