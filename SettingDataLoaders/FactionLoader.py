"""
Faction Loader - Handles loading of faction/organization data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class FactionLoader(WorldDataLoader):
    """Loader for faction data - organized groups"""

    @property
    def data_type(self) -> str:
        return "faction"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'headquarters',
            'goals',
            'leadership',
            'membership'
        ]

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format faction data for embedding"""
        text = f"""
        Faction: {record['name']}

        Type: {record['type']}

        Headquarters: {record['headquarters']}

        Goals: {record['goals']}

        Leadership: {record['leadership']}

        Membership: {record['membership']}
        """

        # Add optional fields if present
        optional_sections = [
            ('ideology', 'Ideology'),
            ('resources', 'Resources'),
            ('enemies', 'Enemies'),
            ('allies', 'Allies'),
            ('roleplay_hooks', 'Roleplay Hooks')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a faction"""
        metadata = {
            'type': 'faction',
            'name': record['name'],
            'faction_type': record['type'],
            'headquarters': record['headquarters'],
            'goals': record['goals'],
            'leadership': record['leadership'],
            'membership': record['membership'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'ideology', 'alignment', 'resources', 'enemies',
            'allies', 'history', 'symbol', 'colors', 'uniform',
            'secret_signs', 'roleplay_hooks', 'influence_level',
            'member_count', 'primary_races', 'territory'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_faction_json(json_path: str = "faction.json"):
    """Create a test faction JSON file if it doesn't exist"""
    test_data = {
        "factions": [
            {
                "name": "The Argent Hand",
                "type": "Religious/Military Order",
                "headquarters": "The Argent Spire fortress-monastery",
                "goals": "Eradicate evil, protect the innocent, uphold divine law",
                "leadership": "Lord Commander Valerius (human paladin)",
                "membership": "500+ members including paladins, clerics, and lay followers from all races",
                "ideology": "Lawful Good, follows the God of Justice and Light",
                "enemies": "Cult of the Sundered Veil, undead, demons",
                "roleplay_hooks": "A young paladin on their first pilgrimage; an inquisitor investigating a noble suspected of necromancy"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test faction data in {json_path}")


if __name__ == "__main__":
    print("=== Testing FactionLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/factions.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_faction_json(json_file)

    # Load and insert data
    loader = FactionLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total factions: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("FactionLoader test completed!")