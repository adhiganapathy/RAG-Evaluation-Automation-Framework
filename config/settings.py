import os
from dataclasses import dataclass
from dotenv import load_dotenv
from framework_core.data_loader import DataLoader

load_dotenv()

@dataclass(frozen=True)
class EvalThresholds:
    faithfulness: float
    answer_relevancy: float
    retrieval_similarity: float
    pii_confidence_cutoff: float
    max_latency_sec: float
    hallucination_max: float
    contextual_precision: float
    contextual_recall: float
    max_response_words: int
    toxicity_max: float
    bias_max: float
    conversation_completeness: float

    @classmethod
    def from_yaml(cls) -> "EvalThresholds":
        raw = DataLoader.load_thresholds()
        return cls(
            faithfulness=raw["rag"]["faithfulness"],
            answer_relevancy=raw["rag"]["answer_relevancy"],
            retrieval_similarity=raw["rag"]["retrieval_similarity"],
            pii_confidence_cutoff=raw["security"]["pii_confidence_cutoff"],
            max_latency_sec=raw["rag"]["max_latency_sec"],
            hallucination_max=raw["rag"]["hallucination_max"],
            contextual_precision=raw["rag"]["contextual_precision"],
            contextual_recall=raw["rag"]["contextual_recall"],
            max_response_words=raw["rag"]["max_response_words"],
            toxicity_max=raw["security"]["toxicity_max"],
            bias_max=raw["security"]["bias_max"],
            conversation_completeness=raw["agent"]["conversation_completeness"],
        )

@dataclass(frozen=True) 
class FrameworkConfig:
    r2r_base_url: str = os.getenv("R2R_BASE_URL", "http://localhost:7272")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    judge_model: str = "openai/gpt-oss-120b"
    kb_filename: str = "company_policy.md"
    thresholds: EvalThresholds = EvalThresholds.from_yaml()