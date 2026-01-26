import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
from pathlib import Path

from FaissVectorDB import FAISSVectorDB  # Changed from get_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a data loading operation"""
    data_type: str
    total_records: int
    successful_inserts: int
    failed_records: List[Dict] = field(default_factory=list)
    inserted_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class WorldDataLoader(ABC):
    """Abstract base class for loading world data into the database"""

    def __init__(
            self,
            json_path: Optional[str] = None,
            adventure_name: str = "vanilla_fantasy",
            adventure_base_path: str = "vanilla_fantasy"
    ):
        """
        Initialize a data loader

        Args:
            json_path: Path to JSON file containing data (optional)
            adventure_name: Name of the adventure for vector database
            adventure_base_path: Base path for JSON files within SettingRawDataJSON
        """
        self.json_path = json_path
        self.adventure_name = adventure_name
        self.adventure_base_path = adventure_base_path

        # Initialize vector database for this adventure
        self.db = FAISSVectorDB(adventure_name)
        self._data_cache: List[Dict] = []

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return the type of data this loader handles"""
        pass

    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """Return list of required fields for this data type"""
        pass

    @abstractmethod
    def format_for_embedding(self, record: Dict[str, Any]) -> str:
        """
        Format a single record into text for embedding

        Args:
            record: A single data record

        Returns:
            Formatted text string
        """
        pass

    @abstractmethod
    def create_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for a single record

        Args:
            record: A single data record

        Returns:
            Metadata dictionary
        """
        pass

    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """
        Helper function to validate a single record

        Args:
            record: Record to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")
            elif not record[field_name]:
                errors.append(f"Empty required field: {field_name}")

        return errors

    def load_from_json(self, json_path: Optional[str] = None) -> List[Dict]:
        """
        Load data from JSON file

        Args:
            json_path: Optional path to override the default

        Returns:
            List of data records
        """
        path = json_path or self.json_path
        if not path:
            # Construct default path using adventure_base_path
            default_path = Path("SettingRawDataJSON") / self.adventure_base_path
            if self.json_filename:
                default_path = default_path / self.json_filename
            path = str(default_path)
            self.json_path = path

        if not os.path.exists(path):
            # Try with .json extension if not already present
            if not path.endswith('.json'):
                path_with_ext = f"{path}.json"
                if os.path.exists(path_with_ext):
                    path = path_with_ext
                else:
                    raise FileNotFoundError(f"JSON file not found: {path} (also tried {path_with_ext})")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract data based on structure
            self._data_cache = self._extract_data(data)

            logger.info(f"Loaded {len(self._data_cache)} {self.data_type} records from {path}")
            return self._data_cache

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")

    @property
    def json_filename(self) -> Optional[str]:
        """
        Get the default JSON filename for this data type.
        Child classes can override this to specify a custom filename.
        """
        return f"{self.data_type}.json"

    def _extract_data(self, json_data: Any) -> List[Dict]:
        """
        Extract the list of records from JSON data.
        This is a simple implementation that child classes can override.

        Default behavior:
        - If JSON is a dict, look for keys matching data_type or data_type + 's'
        - If JSON is a list, return it directly
        """
        if isinstance(json_data, dict):
            # Try singular then plural key
            if self.data_type in json_data:
                data = json_data[self.data_type]
            elif f"{self.data_type}s" in json_data:
                data = json_data[f"{self.data_type}s"]
            else:
                # No matching key, try to see if it's a list of our type
                logger.warning(f"JSON doesn't contain '{self.data_type}' or '{self.data_type}s' key")
                # Check if entire dict is a single record
                if all(field in json_data for field in self.required_fields):
                    return [json_data]
                return []

            # Ensure we return a list
            if isinstance(data, list):
                return data
            else:
                logger.warning(f"Expected list under key, got {type(data).__name__}")
                return []

        elif isinstance(json_data, list):
            # Assume the list contains records of our type
            return json_data
        else:
            logger.error(f"Unexpected JSON structure: {type(json_data).__name__}")
            return []

    def insert_single(self, record: Dict[str, Any]) -> Optional[int]:
        """
        Insert a single record into the database

        Args:
            record: Record to insert

        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Validate
            validation_errors = self.validate_record(record)
            if validation_errors:
                record_name = self._get_record_name(record)
                logger.warning(f"Invalid {self.data_type} record '{record_name}': {', '.join(validation_errors)}")
                return None

            # Format for embedding
            text = self.format_for_embedding(record)
            metadata = self.create_metadata(record)

            # Add adventure metadata
            metadata.update({
                'adventure_name': self.adventure_name,
                'adventure_base_path': self.adventure_base_path,
                'data_type': self.data_type
            })

            # Insert into database
            doc_id = self.db.insert_single(text, metadata)

            record_name = self._get_record_name(metadata)
            logger.debug(f"Inserted {self.data_type}: {record_name} (ID: {doc_id})")

            return doc_id

        except Exception as e:
            record_id = self._get_record_name(record) or 'Unknown'
            logger.error(f"Error inserting {self.data_type} record {record_id}: {e}")
            return None

    def _get_record_name(self, record: Dict[str, Any]) -> str:
        """Extract a name/identifier from a record for logging"""
        # Try common name fields
        for field_name in ['name', 'race_name', 'faction_name', 'creature_name', 'item_name', 'title']:
            if field_name in record:
                return str(record[field_name])

        # Try id fields
        for field_name in ['id', 'record_id', 'entry_id']:
            if field_name in record:
                return f"{self.data_type}_{record[field_name]}"

        return 'Unknown'

    def insert_batch(self, records: List[Dict[str, Any]]) -> LoadResult:
        """
        Insert multiple records into the database

        Args:
            records: List of records to insert
        Returns:
            LoadResult with statistics
        """
        result = LoadResult(
            data_type=self.data_type,
            total_records=len(records),
            successful_inserts=0
        )

        # Prepare data for batch insert
        texts = []
        metadatas = []
        valid_records = []

        for i, record in enumerate(records):
            try:
                # Validate
                validation_errors = self.validate_record(record)
                if validation_errors:
                    record_name = self._get_record_name(record) or f"record_{i}"
                    result.errors.append(f"Invalid record '{record_name}': {', '.join(validation_errors)}")
                    result.failed_records.append(record)
                    continue

                # Format for embedding and metadata
                text = self.format_for_embedding(record)
                metadata = self.create_metadata(record)

                # Add adventure metadata
                metadata.update({
                    'adventure_name': self.adventure_name,
                    'adventure_base_path': self.adventure_base_path,
                    'data_type': self.data_type
                })

                texts.append(text)
                metadatas.append(metadata)
                valid_records.append(record)

            except Exception as e:
                record_name = self._get_record_name(record) or f"record_{i}"
                result.errors.append(f"Error processing record '{record_name}': {str(e)}")
                result.failed_records.append(record)

        # Batch insert valid records
        if texts:
            try:
                inserted_ids = self.db.insert(texts, metadatas)
                result.successful_inserts = len(inserted_ids)
                result.inserted_ids.extend(inserted_ids)

                logger.info(f"Batch inserted {len(inserted_ids)} {self.data_type} records")

            except Exception as e:
                result.errors.append(f"Batch insert failed: {str(e)}")
                # Fallback to individual inserts
                result.successful_inserts = 0
                for i, record in enumerate(valid_records):
                    doc_id = self.insert_single(record)
                    if doc_id is not None:
                        result.successful_inserts += 1
                        result.inserted_ids.append(doc_id)
                    else:
                        result.failed_records.append(record)
                        record_name = self._get_record_name(record) or f"record_{i}"
                        result.errors.append(f"Failed to insert {self.data_type} {record_name}")

        # Count total failed records
        result.failed_records = [r for r in records if r not in valid_records]

        logger.info(f"Inserted {result.successful_inserts}/{result.total_records} {self.data_type} records")
        return result

    def load_and_insert(self, json_path: Optional[str] = None) -> LoadResult:
        """
        Load data from JSON and insert into database

        Args:
            json_path: Optional path to override the default

        Returns:
            LoadResult with statistics
        """
        try:
            # Load data
            records = self.load_from_json(json_path)

            if not records:
                return LoadResult(
                    data_type=self.data_type,
                    total_records=0,
                    successful_inserts=0,
                    errors=["No records loaded from JSON"]
                )

            # Insert data
            result = self.insert_batch(records)

            # Save database
            self.save_database()

            return result

        except Exception as e:
            return LoadResult(
                data_type=self.data_type,
                total_records=0,
                successful_inserts=0,
                errors=[str(e)]
            )

    def save_database(self):
        """Save the database to disk"""
        self.db.save()
        logger.info(f"Database saved for adventure '{self.adventure_name}' after {self.data_type} operations")

    def get_record_template(self) -> Dict[str, Any]:
        """
        Get a template for creating new records

        Returns:
            Dictionary with required fields and examples
        """
        template = {}

        # Add required fields with example values
        for field in self.required_fields:
            # Convert field name to readable example
            example = field.replace('_', ' ').title()
            template[field] = f"Example {example} Value"

        return template

    def get_cached_data(self) -> List[Dict]:
        """
        Get currently cached data

        Returns:
            List of cached records
        """
        return self._data_cache.copy()

    def clear_cache(self):
        """Clear the data cache"""
        self._data_cache = []
        logger.debug(f"Cleared cache for {self.data_type} loader")

    def search_records(self, query: str, k: int = 5, **kwargs) -> List[Dict]:
        """
        Search for records in the database

        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Additional search parameters (filter_metadata, threshold, etc.)

        Returns:
            List of search results
        """
        try:
            results = self.db.search(query, k=k, **kwargs)
            return results
        except Exception as e:
            logger.error(f"Error searching {self.data_type}: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """
        Get statistics for the adventure database

        Returns:
            Database statistics
        """
        try:
            return self.db.get_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def clear_database(self) -> bool:
        """
        Clear all records of this data type from the database

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents of this data type
            all_docs = self.db.get_all_documents()
            docs_to_delete = []

            for doc in all_docs:
                if doc.get('metadata', {}).get('data_type') == self.data_type:
                    docs_to_delete.append(doc['id'])

            if docs_to_delete:
                success = self.db.delete(docs_to_delete)
                if success:
                    logger.info(f"Cleared {len(docs_to_delete)} {self.data_type} records from database")
                    self.db.save()
                return success

            logger.info(f"No {self.data_type} records to clear")
            return True

        except Exception as e:
            logger.error(f"Error clearing {self.data_type} records: {e}")
            return False

    def close(self):
        """Close the database connection"""
        self.db.close()
        logger.info(f"Closed database for adventure '{self.adventure_name}'")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection"""
        self.close()