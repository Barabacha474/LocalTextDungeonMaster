"""
Race Loader - Handles loading of race/species data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class RaceLoader(WorldDataLoader):
    """Loader for race/species data - sentient beings with culture"""

    @property
    def data_type(self) -> str:
        return "race"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'physical_description',
            'homeland',
            'culture',
            'core_traits',
            'roleplay_hooks'
        ]

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format race data for embedding"""
        text = f"""
        Race: {record['name']}
        
        Physical Description: {record['physical_description']}
        
        Homeland: {record['homeland']}
        
        Culture: {record['culture']}
        
        Core Traits: {record['core_traits']}
        
        Roleplay Hooks: {record['roleplay_hooks']}
        """

        # Add optional fields if present
        optional_sections = [
            ('average_lifespan', 'Average Lifespan'),
            ('typical_alignment', 'Typical Alignment'),
            ('language', 'Language'),
            ('special_abilities', 'Special Abilities'),
            ('history', 'History')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a race"""
        metadata = {
            'type': 'race',
            'name': record['name'],
            'category': 'fantasy_world',
            'physical_description': record['physical_description'],
            'homeland': record['homeland'],
            'culture': record['culture'],
            'core_traits': record['core_traits'],
            'roleplay_hooks': record['roleplay_hooks'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'average_lifespan', 'typical_alignment', 'language',
            'special_abilities', 'history', 'size_category',
            'average_height', 'average_weight', 'skin_colors',
            'eye_colors', 'hair_colors', 'diet', 'reproduction',
            'subraces', 'relations', 'governing_body'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_race_json(json_path: str = "race.json"):
    """Create a test race JSON file if it doesn't exist"""
    test_data = {
        "races": [
            {
                "name": "Humans",
                "physical_description": "The most physically diverse race, with heights, builds, and complexions spanning a wide spectrum. They live fast, passionate lives.",
                "homeland": "Heartland Kingdoms — feudal states of castles, bustling trade cities, and fertile farms.",
                "culture": "Their societies are defined by ambition, innovation, and short memories, leading to rapidly shifting politics and borders.",
                "core_traits": "Versatile & Ambitious: Their strength is their relentless drive and ability to master any skill given time. Short-Lived Zeal: They build great empires and forge deep enmities within a single lifetime.",
                "roleplay_hooks": "A human knight sworn to a dying kingdom; a merchant prince funding an expedition; a scholar seeking lost knowledge to gain an edge.",
                "average_lifespan": "80-100 years",
                "typical_alignment": "Any, but often Lawful or Neutral",
                "language": "Common",
                "size_category": "Medium"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test race data in {json_path}")


if __name__ == "__main__":
    print("=== Testing RaceLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/races.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_race_json(json_file)

    # Load and insert data
    loader = RaceLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total races: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("RaceLoader test completed!")