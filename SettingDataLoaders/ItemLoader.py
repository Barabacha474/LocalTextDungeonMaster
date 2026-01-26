"""
Item Loader - Handles loading of item/artifact/technology data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class ItemLoader(WorldDataLoader):
    """Loader for item data - objects/technology/artifacts"""

    @property
    def data_type(self) -> str:
        return "item"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'description',
            'properties',
            'rarity',
            'purpose'
        ]

    def _extract_data(self, json_data: Any) -> List[Dict]:
        """Override to handle multiple possible JSON keys for item data"""
        if isinstance(json_data, dict):
            # Try various keys for item data
            possible_keys = ['item', 'items', 'artifact', 'artifacts', 'technology', 'technologies']
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
        """Format item data for embedding"""
        text = f"""
        Item: {record['name']}

        Type: {record['type']}

        Description: {record['description']}

        Properties: {record['properties']}

        Rarity: {record['rarity']}

        Purpose: {record['purpose']}
        """

        # Add optional fields if present
        optional_sections = [
            ('history', 'History'),
            ('crafting', 'Crafting Process'),
            ('activation', 'Activation Method'),
            ('limitations', 'Limitations'),
            ('value', 'Value')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for an item"""
        metadata = {
            'type': 'item',
            'name': record['name'],
            'item_type': record['type'],
            'description': record['description'],
            'properties': record['properties'],
            'rarity': record['rarity'],
            'purpose': record['purpose'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'history', 'crafting', 'materials', 'weight',
            'dimensions', 'value', 'activation', 'duration',
            'range', 'charges', 'attunement', 'curse',
            'creator', 'current_owner', 'location', 'quest_hooks'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_item_json(json_path: str = "item.json"):
    """Create a test item JSON file if it doesn't exist"""
    test_data = {
        "items": [
            {
                "name": "Moon-touched Silver Sword",
                "type": "Magical Weapon",
                "description": "A beautifully crafted silver longsword that glows with a soft moonlight when drawn.",
                "properties": "+1 to hit and damage against shadow creatures and werewolves, glows in darkness",
                "rarity": "Rare",
                "purpose": "Combat against supernatural creatures, ceremonial purposes",
                "history": "Forged by elven smiths during the War of Shadows, passed down through generations of monster hunters",
                "value": "5000 gold pieces",
                "activation": "Automatically glows when undead or shadow creatures are nearby",
                "limitations": "Silver is softer than steel, requires special maintenance"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test item data in {json_path}")


if __name__ == "__main__":
    print("=== Testing ItemLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/items.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_item_json(json_file)

    # Load and insert data
    loader = ItemLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total items: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("ItemLoader test completed!")