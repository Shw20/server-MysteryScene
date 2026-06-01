import json
from pathlib import Path
from typing import Any

from .evidence_schema import EvidenceDataset
from .scenario_schema import ScenarioDataset


def load_raw_json(file_path: str | Path) -> Any:
    """UTF-8 JSON 파일을 원본 dict/list 형태로 읽는다."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_evidence_dataset(file_path: str | Path) -> EvidenceDataset:
    """단일 증거 데이터셋 파일을 Pydantic 모델로 검증해 읽는다."""
    raw_data = load_raw_json(file_path)
    return EvidenceDataset.parse_obj(raw_data)


def load_scenario_dataset(file_path: str | Path) -> ScenarioDataset:
    """사건 시나리오 파일을 Pydantic 모델로 검증해 읽는다."""
    raw_data = load_raw_json(file_path)
    return ScenarioDataset.parse_obj(raw_data)


def load_evidence_items(file_path: str | Path) -> list[dict]:
    """RAG 인덱싱에 필요한 evidence 목록만 추출한다.

    예전 증거 전용 JSON과 현재 시나리오 JSON을 둘 다 받을 수 있게 유지한다.
    """

    raw_data = load_raw_json(file_path)
    if isinstance(raw_data, dict) and "overview" in raw_data and "solution" in raw_data:
        dataset = ScenarioDataset.parse_obj(raw_data)
    else:
        dataset = EvidenceDataset.parse_obj(raw_data)
    return [item.dict() for item in dataset.evidence]
