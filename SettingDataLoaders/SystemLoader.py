"""
System Loader - Handles loading of system/magic/tech data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class SystemLoader(WorldDataLoader):
    """Loader for system data - power/magic/tech systems"""

    @property
    def data_type(self) -> str:
        return "system"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'description',
            'rules',
            'source',
            'limitations'
        ]

    def _extract_data(self, json_data: Any) -> List[Dict]:
        """Override to handle multiple possible JSON keys for system data"""
        if isinstance(json_data, dict):
            # Try various keys for system data
            possible_keys = ['system', 'systems', 'magic', 'magics', 'technology', 'technologies']
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
        """Format system data for embedding"""
        text = f"""
        System: {record['name']}

        Type: {record['type']}

        Description: {record['description']}

        Rules: {record['rules']}

        Source: {record['source']}

        Limitations: {record['limitations']}
        """

        # Add optional fields if present
        optional_sections = [
            ('components', 'Components'),
            ('dangers', 'Dangers'),
            ('learning', 'Learning Process'),
            ('history', 'History'),
            ('variations', 'Variations')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a system"""
        metadata = {
            'type': 'system',
            'name': record['name'],
            'system_type': record['type'],
            'description': record['description'],
            'rules': record['rules'],
            'source': record['source'],
            'limitations': record['limitations'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'components', 'dangers', 'learning', 'history',
            'variations', 'cost', 'casting_time', 'range',
            'duration', 'gestures', 'incantations', 'mana_cost',
            'prerequisites', 'effects', 'users', 'discovery_date'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_system_json(json_path: str = "system.json"):
    """Create a test system JSON file if it doesn't exist"""
    test_data = {
        "systems": [
            {
                "name": "Arcane Weave",
                "type": "Magic System",
                "description": "Intellectual, formulaic magic that imposes the caster's will upon the Aether through precise gestures, runes, and incantations.",
                "rules": "Requires verbal, somatic, and material components. Limited by caster's knowledge and mental endurance.",
                "source": "Study and intellect",
                "limitations": "Aether Burn from overcasting, requires years of study, cannot create permanent life",
                "components": "Verbal (incantations), Somatic (gestures), Material (focus items like wands or rare components)",
                "dangers": "Backlash from failed spells, mental exhaustion, attracting unwanted attention from otherworldly beings",
                "history": "Developed by ancient elves and refined by human wizards over millennia"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test system data in {json_path}")


if __name__ == "__main__":
    print("=== Testing SystemLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/systems.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_system_json(json_file)

    # Load and insert data
    loader = SystemLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total systems: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("SystemLoader test completed!")