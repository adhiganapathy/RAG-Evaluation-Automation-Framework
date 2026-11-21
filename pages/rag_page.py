from dataclasses import dataclass, field
from shared.abstractions.exception import R2RException
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from framework_core.base_endpoint import BaseEndpoint
import time


@dataclass
class RAGResultDTO:
    query: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)
    latency_sec: float = 0.0


class RAGPage(BaseEndpoint):
    """Page Object for R2R Document Ingestion & RAG Retrieval/Generation."""
    PII_ENTITIES = ["PHONE_NUMBER", "EMAIL_ADDRESS", "IN_PAN", "CREDIT_CARD"]
    def __init__(self, client, enable_pii_guardrail: bool = True):
        super().__init__(client)
        self._enable_pii_guardrail = enable_pii_guardrail
        self._analyzer = AnalyzerEngine()
        pan_pattern = Pattern(name="in_pan_pattern", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", score=0.85)
        self._analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="IN_PAN", patterns=[pan_pattern])
        )
    def ensure_document_ingested(self, file_path: str, title: str) -> None:
        existing_docs = self._client.documents.list().get("results", [])
        if any(doc.get("title") == title for doc in existing_docs):
            return
        try:
            self._client.documents.create(file_path=file_path)
        except R2RException as exc:
            if "already exists" not in str(exc):
                raise

    def execute(self, query: str) -> RAGResultDTO:
        start_time = time.perf_counter()
        raw_res = self._client.retrieval.rag(
            query=query,
            search_settings=self.default_search_settings,
        )
        elapsed_sec = round(time.perf_counter() - start_time, 3)
        result = self._parse_rag_response(query, raw_res)
        result.latency_sec = elapsed_sec
        return result

    def _parse_rag_response(self, query: str, raw_res: dict) -> RAGResultDTO:
        results = raw_res.get("results", {})
        chunks = results.get("search_results", {}).get("chunk_search_results", [])
        raw_answer = results.get("completion", "")
        safe_answer = self._apply_pii_guardrail(raw_answer)

        return RAGResultDTO(
            query=query,
            answer=safe_answer,
            contexts=[c.get("text", "") for c in chunks if c.get("text")],
            similarity_scores=[c.get("score", 0.0) for c in chunks]
        )
    def _apply_pii_guardrail(self, answer: str) -> str:
        if not self._enable_pii_guardrail:
            return answer
        findings = self._analyzer.analyze(
            text=answer,
            entities=self.PII_ENTITIES,
            language="en"
        )
        if any(f.score >= 0.70 for f in findings):
            return "I cannot provide this answer as it contains confidential or personally identifiable information."
        return answer

        