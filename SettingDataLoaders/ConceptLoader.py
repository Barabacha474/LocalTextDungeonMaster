"""
Concept Loader - Handles loading of abstract concept data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class ConceptLoader(WorldDataLoader):
    """Loader for concept data - abstract world elements"""

    @property
    def data_type(self) -> str:
        return "concept"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'description',
            'significance',
            'manifestation',
            'rules'
        ]

    def _extract_data(self, json_data: Any) -> List[Dict]:
        """Override to handle multiple possible JSON keys for concept data"""
        if isinstance(json_data, dict):
            # Try various keys for concept data
            possible_keys = ['concept', 'concepts', 'law', 'laws', 'principle', 'principles']
            for key in possible_keys:
                if key in json_data:
                    data = json_data[key]
                    if isinstance(data, list):
                        return data
            # No matching key found
            return []
        elif isinstance(json_data, list):
            return json_data
        else:
            return []

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format concept data for embedding"""
        text = f"""
        Concept: {record['name']}

        Type: {record['type']}

        Description: {record['description']}

        Significance: {record['significance']}

        Manifestation: {record['manifestation']}

        Rules: {record['rules']}
        """

        # Add optional fields if present
        optional_sections = [
            ('examples', 'Examples'),
            ('exceptions', 'Exceptions'),
            ('history', 'Historical Understanding'),
            ('debates', 'Philosophical Debates'),
            ('related_concepts', 'Related Concepts')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a concept"""
        metadata = {
            'type': 'concept',
            'name': record['name'],
            'concept_type': record['type'],
            'description': record['description'],
            'significance': record['significance'],
            'manifestation': record['manifestation'],
            'rules': record['rules'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'examples', 'exceptions', 'history', 'debates',
            'related_concepts', 'discoverer', 'discovery_date',
            'proofs', 'counterarguments', 'practical_applications',
            'cultural_impact', 'religious_significance'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_concept_json(json_path: str = "concept.json"):
    """Create a test concept JSON file if it doesn't exist"""
    test_data = {
        "concepts": [
            {
                "name": "The Aether",
                "type": "Cosmological Principle",
                "description": "The raw potential of creation and change that flows through all things. The fundamental substance of magic.",
                "significance": "Explains the existence and function of magic, connects all living things, determines the laws of reality",
                "manifestation": "Visible as ley lines, magical auras, spontaneous magical phenomena, and in places of power",
                "rules": "Cannot be created or destroyed, only transformed; follows conservation principles; responds to will and emotion",
                "examples": "Magic spells, enchanted items, magical creatures, ley line convergences",
                "history": "First theorized by elven sages 2000 years ago, proven by dwarven geomancers 500 years ago",
                "practical_applications": "Magic casting, enchanting, divination, teleportation, healing"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test concept data in {json_path}")


if __name__ == "__main__":
    print("=== Testing ConceptLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/concepts.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_concept_json(json_file)

    # Load and insert data
    loader = ConceptLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total concepts: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("ConceptLoader test completed!")