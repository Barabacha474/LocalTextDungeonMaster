"""
Creature Loader - Handles loading of creature/monster data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class CreatureLoader(WorldDataLoader):
    """Loader for creature data - non-sentient beings"""

    @property
    def data_type(self) -> str:
        return "creature"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'classification',
            'description',
            'habitat',
            'behavior',
            'diet'
        ]

    def _extract_data(self, json_data: Any) -> List[Dict]:
        """Override to handle multiple possible JSON keys for creature data"""
        if isinstance(json_data, dict):
            # Try various keys for creature data
            possible_keys = ['creature', 'creatures', 'monster', 'monsters', 'beast', 'beasts', 'animal', 'animals']
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
        """Format creature data for embedding"""
        text = f"""
        Creature: {record['name']}

        Classification: {record['classification']}

        Description: {record['description']}

        Habitat: {record['habitat']}

        Behavior: {record['behavior']}

        Diet: {record['diet']}
        """

        # Add optional fields if present
        optional_sections = [
            ('abilities', 'Special Abilities'),
            ('weaknesses', 'Weaknesses'),
            ('threat_level', 'Threat Level'),
            ('interaction', 'Interaction with Others'),
            ('loot', 'Loot/Resources')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a creature"""
        metadata = {
            'type': 'creature',
            'name': record['name'],
            'classification': record['classification'],
            'description': record['description'],
            'habitat': record['habitat'],
            'behavior': record['behavior'],
            'diet': record['diet'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'abilities', 'weaknesses', 'threat_level', 'size',
            'intelligence', 'speed', 'armor', 'hit_points',
            'attack', 'defense', 'loot', 'reproduction',
            'lifespan', 'senses', 'communication', 'social_structure'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_creature_json(json_path: str = "creature.json"):
    """Create a test creature JSON file if it doesn't exist"""
    test_data = {
        "creatures": [
            {
                "name": "Arcane Fox (Kitsunari)",
                "classification": "Aether-Touched Beast",
                "description": "A fox with two tails and fur that shifts like a mirage. Mildly telepathic and capable of minor illusions.",
                "habitat": "Ley Line verges, magical groves, enchanted forests",
                "behavior": "Curious, playful trickster, protective of magical areas, avoids direct confrontation",
                "diet": "Omnivorous - berries, small animals, magical energy from ley lines",
                "abilities": "Telepathic communication, minor illusion spells, short-range phase shifting, can sense magic",
                "weaknesses": "Vulnerable to cold iron, dislikes loud noises, cannot phase through consecrated ground",
                "threat_level": "Low (non-aggressive unless threatened)",
                "loot": "Arcane Fox Tail Feather (rare alchemical ingredient), Kitsunari Pelt (magical component)"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test creature data in {json_path}")


if __name__ == "__main__":
    print("=== Testing CreatureLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/creatures.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_creature_json(json_file)

    # Load and insert data
    loader = CreatureLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total creatures: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("CreatureLoader test completed!")