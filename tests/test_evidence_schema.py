import unittest
from pathlib import Path

from pydantic import ValidationError

from src.database.evidence_loader import load_evidence_dataset, load_evidence_items
from src.database.evidence_schema import EvidenceDataset
from src.database.scenario_schema import ScenarioDataset
from src.rag.ingestion.normalizer import normalize_evidence


def build_valid_dataset() -> dict:
    """스키마/정규화 테스트에 쓰는 최소 정상 증거 데이터."""
    return {
        "schema_version": "2.0",
        "scenario_id": "case001",
        "title": "Test Case",
        "language": "ko",
        "evidence": [
            {
                "id": "E01",
                "title": "Coffee Cup",
                "level": 1,
                "category": "physical_evidence",
                "content": "A drug was detected in the coffee cup.",
                "summary": "Coffee cup has drug residue.",
                "keywords": ["coffee", "drug"],
                "tags": ["toxicology"],
                "entities": [
                    {
                        "name": "coffee cup",
                        "role": "object",
                        "aliases": ["cup", "mug"],
                    }
                ],
                "source_type": "lab_report",
                "image_url": "/assets/evidence/cup.jpg",
                "image_alt": "Coffee cup photo",
                "related_ids": ["E02"],
                "contradicts": [],
                "visibility": "public",
                "search_text": "sleeping drug residue",
            },
            {
                "id": "E02",
                "title": "Library Alibi",
                "level": 2,
                "category": "testimony",
                "content": "The suspect claimed to be in the library.",
                "keywords": ["alibi", "library"],
                "tags": [],
                "entities": [],
                "source_type": "statement",
                "related_ids": ["E01"],
                "contradicts": [],
                "visibility": "locked",
            },
        ],
    }


class EvidenceSchemaTests(unittest.TestCase):
    """증거/시나리오 JSON이 깨진 참조를 배포 전에 잡는지 검증한다."""

    def test_valid_dataset_can_be_normalized_for_rag(self) -> None:
        dataset = EvidenceDataset.parse_obj(build_valid_dataset())

        chunks = normalize_evidence(dataset.dict())

        self.assertEqual(len(chunks), 2)
        self.assertEqual(dataset.evidence[0].image_url, "/assets/evidence/cup.jpg")
        self.assertEqual(dataset.evidence[0].image_alt, "Coffee cup photo")
        self.assertEqual(chunks[0].metadata.scenario_id, "case001")
        self.assertIn("coffee", chunks[0].metadata.tags)
        self.assertIn("mug", chunks[0].metadata.tags)
        self.assertIn("sleeping drug residue", chunks[0].content)

    def test_duplicate_ids_are_rejected(self) -> None:
        data = build_valid_dataset()
        data["evidence"][1]["id"] = "E01"

        with self.assertRaises(ValidationError):
            EvidenceDataset.parse_obj(data)

    def test_unknown_references_are_rejected(self) -> None:
        data = build_valid_dataset()
        data["evidence"][0]["related_ids"] = ["E99"]

        with self.assertRaises(ValidationError):
            EvidenceDataset.parse_obj(data)

    def test_example_file_matches_v2_schema(self) -> None:
        example_path = Path("src/database/evidence_v2.example.json")

        dataset = load_evidence_dataset(example_path)

        self.assertEqual(dataset.schema_version, "2.0")
        self.assertEqual(dataset.scenario_id, "case001")
        self.assertGreater(len(dataset.evidence), 0)

    def test_scenario_example_file_matches_v2_schema(self) -> None:
        example_path = Path("src/database/scenario_v2.example.json")

        raw_items = load_evidence_items(example_path)
        dataset = ScenarioDataset.parse_file(example_path)
        chunks = normalize_evidence(dataset.dict())

        self.assertEqual(dataset.schema_version, "2.0")
        self.assertEqual(dataset.scenario_id, "case112_worlds_most_high_maintenance_friend")
        self.assertEqual(len(raw_items), len(dataset.evidence))
        self.assertEqual(len(chunks), len(dataset.evidence))
        self.assertIn("기네스", chunks[1].metadata.tags)

    def test_scenario_unknown_location_reference_is_rejected(self) -> None:
        data = build_valid_dataset()
        scenario = {
            "schema_version": "2.0",
            "scenario_id": "case001",
            "title": "Test Case",
            "language": "ko",
            "difficulty": 2,
            "overview": {
                "player_intro": "intro",
                "internal_summary": "summary",
            },
            "characters": [
                {
                    "id": "C01",
                    "name": "Detective",
                    "role": "detective",
                }
            ],
            "locations": [
                {
                    "id": "L01",
                    "name": "Study",
                }
            ],
            "evidence": data["evidence"],
            "solution": {
                "culprit_character_id": "C01",
                "motive": "test",
                "method": "test",
                "decisive_evidence_ids": ["E01"],
                "explanation": "test",
            },
        }
        scenario["evidence"][0]["location_id"] = "L99"

        with self.assertRaises(ValidationError):
            ScenarioDataset.parse_obj(scenario)
