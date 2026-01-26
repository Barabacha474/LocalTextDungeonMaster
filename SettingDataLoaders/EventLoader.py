"""
Event Loader - Handles loading of historical event data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class EventLoader(WorldDataLoader):
    """Loader for event data - historical occurrences"""

    @property
    def data_type(self) -> str:
        return "event"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'description',
            'time_period',
            'key_figures',
            'consequences'
        ]

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format event data for embedding"""
        text = f"""
        Event: {record['name']}

        Type: {record['type']}

        Description: {record['description']}

        Time Period: {record['time_period']}

        Key Figures: {record['key_figures']}

        Consequences: {record['consequences']}
        """

        # Add optional fields if present
        optional_sections = [
            ('causes', 'Causes'),
            ('locations', 'Locations'),
            ('artifacts', 'Related Artifacts'),
            ('legacy', 'Legacy'),
            ('lessons', "Historical Lessons")
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for an event"""
        metadata = {
            'type': 'event',
            'name': record['name'],
            'event_type': record['type'],
            'description': record['description'],
            'time_period': record['time_period'],
            'key_figures': record['key_figures'],
            'consequences': record['consequences'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'causes', 'locations', 'duration', 'casualties',
            'artifacts', 'winners', 'losers', 'treaties',
            'legacy', 'lessons', 'historical_sources',
            'current_interpretations', 'anniversaries'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_event_json(json_path: str = "event.json"):
    """Create a test event JSON file if it doesn't exist"""
    test_data = {
        "events": [
            {
                "name": "The Sundering",
                "type": "Magical Cataclysm",
                "description": "A catastrophic event where an archmage attempted to tear the veil between worlds, causing reality to fracture in certain areas.",
                "time_period": "300 years ago, during the Age of Arcanum",
                "key_figures": "Archmage Malakar (the instigator), High Priestess Elara (who helped contain it), The Silent Council (who passed judgment)",
                "consequences": "Creation of the Blighted Marshes, formation of the Starfall Crater, establishment of the Guild of Arcane Concordance to regulate magic",
                "causes": "Malakar's obsession with contacting eldritch beings, misuse of ancient artifacts, failure of magical safeguards",
                "legacy": "Magic is now heavily regulated, fear of arcane catastrophes persists, some areas remain dangerously unstable"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test event data in {json_path}")


if __name__ == "__main__":
    print("=== Testing EventLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/events.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_event_json(json_file)

    # Load and insert data
    loader = EventLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total events: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("EventLoader test completed!")