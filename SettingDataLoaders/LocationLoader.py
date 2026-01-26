"""
Location Loader - Handles loading of location data
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from AbstractParentLoader import WorldDataLoader, logger

logger = logging.getLogger(__name__)


class LocationLoader(WorldDataLoader):
    """Loader for location data - geographical features"""

    @property
    def data_type(self) -> str:
        return "location"

    @property
    def required_fields(self) -> List[str]:
        return [
            'name',
            'type',
            'description',
            'region',
            'notable_features'
        ]

    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """Format location data for embedding"""
        text = f"""
        Location: {record['name']}

        Type: {record['type']}

        Description: {record['description']}

        Region: {record['region']}

        Notable Features: {record['notable_features']}
        """

        # Add optional fields if present
        optional_sections = [
            ('inhabitants', 'Inhabitants'),
            ('history', 'History'),
            ('dangers', 'Dangers'),
            ('resources', 'Resources'),
            ('climate', 'Climate')
        ]

        for field, label in optional_sections:
            if field in record and record[field]:
                text += f"\n\n{label}: {record[field]}"

        return text

    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for a location"""
        metadata = {
            'type': 'location',
            'name': record['name'],
            'location_type': record['type'],
            'description': record['description'],
            'region': record['region'],
            'notable_features': record['notable_features'],
            'added_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = [
            'inhabitants', 'population', 'governing_faction',
            'history', 'dangers', 'resources', 'climate',
            'architecture', 'defenses', 'secret_areas',
            'trade_routes', 'festivals', 'local_legends',
            'size', 'coordinates', 'travel_time'
        ]

        for field in optional_fields:
            if field in record and record[field]:
                metadata[field] = record[field]

        return metadata


def create_test_location_json(json_path: str = "location.json"):
    """Create a test location JSON file if it doesn't exist"""
    test_data = {
        "locations": [
            {
                "name": "Elderwood Village",
                "type": "Rural Settlement",
                "description": "A peaceful village nestled in a mystical forest, surrounded by ancient oak trees that glow faintly at night.",
                "region": "The Whispering Woods",
                "notable_features": "The Whispering Glade (sacred clearing), Moonfall Inn, Crystal River with healing properties",
                "inhabitants": "500 villagers (mostly humans and halflings), some forest spirits",
                "dangers": "Forest predators, occasional bandit raids, mischievous sprites",
                "climate": "Temperate forest climate, mild winters, rainy springs"
            }
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Created test location data in {json_path}")


if __name__ == "__main__":
    print("=== Testing LocationLoader ===")

    json_file = "../SettingRawDataJSON/vanilla_fantasy/locations.json"

    # Create test JSON if it doesn't exist
    if not os.path.exists(json_file):
        create_test_location_json(json_file)

    # Load and insert data
    loader = LocationLoader(json_file)
    result = loader.load_and_insert()

    # Print results
    print(f"Total locations: {result.total_records}")
    print(f"Successfully inserted: {result.successful_inserts}")

    if result.errors:
        print(f"Errors: {result.errors}")

    print("LocationLoader test completed!")