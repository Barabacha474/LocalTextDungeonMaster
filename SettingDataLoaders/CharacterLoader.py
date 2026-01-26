"""
Character Loader - Handles loading of character/NPC data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class CharacterLoader(WorldDataLoader):
    """Loader for character data - individual persons/NPCs"""

    @property
    def data_type(self) -> str:
        return "character"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'race',
            'occupation',
            'personality',
            'background',
            'motivations'
        ]

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format character data for embedding"""
        text = f"""
        Character: {record['name']}

        Race: {record['race']}

        Occupation: {record['occupation']}

        Personality: {record['personality']}

        Background: {record['background']}

        Motivations: {record['motivations']}
        """

        # Add optional fields if present
        optional_sections = [
            ('appearance', 'Appearance'),
            ('skills_abilities', 'Skills & Abilities'),
            ('relationships', 'Relationships'),
            ('secrets', 'Secrets'),
            ('voice_quirks', 'Voice & Quirks')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a character"""
        metadata = {
            'type': 'character',
            'name': record['name'],
            'race': record['race'],
            'occupation': record['occupation'],
            'personality': record['personality'],
            'background': record['background'],
            'motivations': record['motivations'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'appearance', 'age', 'gender', 'alignment',
            'skills_abilities', 'inventory', 'relationships',
            'secrets', 'voice_quirks', 'faction', 'location',
            'quest_hooks', 'combat_stats', 'dialogue_examples'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_character_json(json_path: str = "character.json"):
    """Create a test character JSON file if it doesn't exist"""
    test_data = {
        "characters": [
            {
                "name": "Borin Ironheart",
                "race": "Dwarf",
                "occupation": "Master Blacksmith",
                "personality": "Stubborn but honorable, speaks in short, gruff sentences. Values craftsmanship above all else.",
                "background": "Former thane of a lost mountain hold, now works as a blacksmith in a human city.",
                "motivations": "To reclaim his family's ancestral forge from the goblin infestation.",
                "appearance": "Stocky with a braided red beard, covered in soot and burn scars. Wears a leather apron.",
                "age": "150 years",
                "skills_abilities": "Master blacksmith, skilled with warhammer, can identify any metal by smell."
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test character data in {json_path}")


if __name__ == "__main__":
    print("=== Testing CharacterLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/characters.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_character_json(json_file)

    # Load and insert data
    loader = CharacterLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total characters: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("CharacterLoader test completed!")