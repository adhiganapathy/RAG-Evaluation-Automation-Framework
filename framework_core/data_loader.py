import json
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

class DataLoader:
    """Data Provider utility decoupling external JSON/YAML datasets from test code."""

    @staticmethod
    def load_json(filename: str) -> list[dict]:
        file_path = DATA_DIR / filename
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_thresholds() -> dict:
        with open(CONFIG_DIR / "thresholds.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_kb_document_path(filename: str) -> str:
        return str(DATA_DIR / "knowledge_base" / filename)