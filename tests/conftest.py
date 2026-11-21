import pytest
from r2r import R2RClient
from config.settings import FrameworkConfig
from framework_core.data_loader import DataLoader
from framework_core.llm_judge import GroqJudgeAdapter
from pages.rag_page import RAGPage
from pages.agent_page import AgentPage
from evaluators.rag_evaluator import RAGEvaluator
from evaluators.security_evaluator import SecurityEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from evaluators.safety_evaluator import SafetyEvaluator
from evaluators.agent_evaluator import AgentEvaluator


def pytest_html_report_title(report):
    report.title = "GenAI RAG & Agent Evaluation Report (Data-Driven POM)"


def pytest_html_results_table_header(cells):
    cells.insert(1, "<th>Test Case Name</th>")
    cells.insert(2, "<th>Suite Category</th>")
    cells.insert(3, "<th>Evaluation Metrics & Verdict</th>")


def pytest_html_results_table_row(report, cells):
    cells.insert(1, f"<td><b>{getattr(report, 'tc_name', report.nodeid)}</b></td>")
    cells.insert(2, f"<td>{getattr(report, 'tc_category', 'General')}</td>")
    cells.insert(3, f"<td>{getattr(report, 'tc_metrics', '-')}</td>")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    funcargs = getattr(item, "funcargs", {})
    if "case" in funcargs:
        case = funcargs["case"]
        desc = case.get("description") or case.get("attack_category", "")
        report.tc_name = f"{case['test_id']}: {desc}"
        report.tc_category = "AI Security Red-Team" if "attack_category" in case else "RAG Grounding & Faithfulness"
    elif "scenario" in funcargs:
        scenario = funcargs["scenario"]
        report.tc_name = f"{scenario['test_id']}: {scenario['scenario']}"
        report.tc_category = "Multi-Turn Agentic Memory"

    report.tc_metrics = getattr(item, "eval_summary", "Executed")


@pytest.fixture(scope="session")
def config() -> FrameworkConfig:
    return FrameworkConfig()


@pytest.fixture(scope="session")
def r2r_client(config: FrameworkConfig) -> R2RClient:
    return R2RClient(base_url=config.r2r_base_url)


@pytest.fixture(scope="session")
def rag_page(r2r_client: R2RClient, config: FrameworkConfig) -> RAGPage:
    page = RAGPage(r2r_client)
    assert page.health_check(), "R2R server unreachable"
    doc_path = DataLoader.get_kb_document_path(config.kb_filename)
    page.ensure_document_ingested(doc_path, config.kb_filename)
    return page


@pytest.fixture(scope="function")
def agent_page(r2r_client: R2RClient) -> AgentPage:
    page = AgentPage(r2r_client)
    page.reset_session()
    return page


@pytest.fixture(scope="session")
def rag_evaluator(config: FrameworkConfig) -> RAGEvaluator:
    judge = GroqJudgeAdapter(api_key=config.groq_api_key, model_name=config.judge_model)
    return RAGEvaluator(judge_llm=judge, thresholds=config.thresholds)


@pytest.fixture(scope="session")
def security_evaluator(config: FrameworkConfig) -> SecurityEvaluator:
    return SecurityEvaluator(
        groq_api_key=config.groq_api_key,
        judge_model=config.judge_model,
        confidence_cutoff=config.thresholds.pii_confidence_cutoff,
    )


@pytest.fixture(scope="session")
def performance_evaluator(config: FrameworkConfig) -> PerformanceEvaluator:
    return PerformanceEvaluator(
        max_latency_sec=config.thresholds.max_latency_sec,
        max_response_words=config.thresholds.max_response_words,
    )


@pytest.fixture(scope="session")
def safety_evaluator(config: FrameworkConfig) -> SafetyEvaluator:
    judge = GroqJudgeAdapter(api_key=config.groq_api_key, model_name=config.judge_model)
    return SafetyEvaluator(judge_llm=judge, thresholds=config.thresholds)


@pytest.fixture(scope="session")
def agent_evaluator(config: FrameworkConfig) -> AgentEvaluator:
    return AgentEvaluator(
        groq_api_key=config.groq_api_key,
        judge_model=config.judge_model,
        min_score=config.thresholds.conversation_completeness,
    )