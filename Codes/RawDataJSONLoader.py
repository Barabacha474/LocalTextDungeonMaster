import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from datetime import datetime

from FaissVectorDB import FAISSVectorDB


class UniversalAdventureLoader:
    """
    Loads every JSON file from an adventure's raw data folder into its Faiss DB.
    No assumptions about file names or content structure – everything becomes a vector.
    """

    BATCH_SIZE = 10000  # Batch size for Faiss insertions

    def __init__(self, adventure_name: str, debug: bool = False):
        """
        Args:
            adventure_name: Name of the adventure (folder under SettingRawDataJSON/)
            debug: If True, print detailed progress information.
        """
        self.adventure_name = adventure_name
        self.debug = debug
        self.raw_root = Path("../SettingRawDataJSON") / adventure_name
        self.db = FAISSVectorDB(adventure_name)

    def run(self):
        """Main entry point: find all JSON files and insert all records."""
        if not self.raw_root.exists():
            print(f"❌ Adventure folder not found: {self.raw_root}")
            return

        # Recursively collect all .json files
        json_files = [
            f for f in self.raw_root.rglob("*.json")
            if f.name != "PromptCore.json"
        ]

        if not json_files:
            print(f"⚠️  No JSON files found under {self.raw_root}")
            return

        if self.debug:
            print(f"🔍 DEBUG: Found {len(json_files)} JSON files:\n'PromptCore.json' is ALWAYS ignored.")
            for f in json_files:
                rel_path = f.relative_to(self.raw_root.parent.parent)
                print(f"   📄 {rel_path}")

        total_records = 0
        total_inserted = 0
        batch_texts: List[str] = []
        batch_metadatas: List[Dict] = []

        for json_path in json_files:
            file_records = 0
            file_inserted = 0

            # Extract all record-like dictionaries from this file
            for record in self._extract_records_from_file(json_path):
                total_records += 1
                file_records += 1
                try:
                    text = self._record_to_text(record)
                    metadata = self._record_to_metadata(record, json_path)

                    batch_texts.append(text)
                    batch_metadatas.append(metadata)

                    # Flush batch if size reached
                    if len(batch_texts) >= self.BATCH_SIZE:
                        inserted = self._insert_batch(batch_texts, batch_metadatas)
                        total_inserted += inserted
                        file_inserted += inserted
                        batch_texts.clear()
                        batch_metadatas.clear()

                except Exception as e:
                    if self.debug:
                        print(f"   ⚠️  DEBUG: Failed to process record: {e}")

            # Insert remaining records for this file
            if batch_texts:
                inserted = self._insert_batch(batch_texts, batch_metadatas)
                total_inserted += inserted
                file_inserted += inserted
                batch_texts.clear()
                batch_metadatas.clear()

            if self.debug and file_records > 0:
                rel_path = json_path.relative_to(self.raw_root.parent.parent)
                print(f"   📊 {rel_path}: {file_records} records, {file_inserted} inserted")

        print("\n" + "=" * 60)
        print(f"✅ Load complete for adventure '{self.adventure_name}'")
        print(f"   Total records extracted : {total_records}")
        print(f"   Successfully inserted    : {total_inserted}")
        print(f"   Failed / skipped         : {total_records - total_inserted}")
        self.db.save()
        self.db.close()

    # ----------------------------------------------------------------------
    # JSON traversal – extract every dictionary that is likely a "record"
    # ----------------------------------------------------------------------
    def _extract_records_from_file(self, json_path: Path) -> Iterator[Dict]:
        """
        Yield every dictionary that should be treated as a standalone record.
        Works with:
        - Root-level list of dicts
        - Root dict that is itself a record
        - Root dict containing one or more lists of dicts (e.g., {"characters": [...]})
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"   ❌ Cannot read {json_path.name}: {e}")
            return

        # Case 1: Root is a list -> each element is a record
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        # Case 2: Root is a dict
        if isinstance(data, dict):
            # If this dict itself looks like a record, yield it
            if self._looks_like_record(data):
                yield data

            # Also look for any list-of-dicts values inside the dict
            for key, value in data.items():
                if isinstance(value, list):
                    for subitem in value:
                        if isinstance(subitem, dict):
                            yield subitem

    @staticmethod
    def _looks_like_record(obj: Dict) -> bool:
        """Heuristic: a dict with at least one of these fields is probably a record."""
        content_fields = {"name", "id", "title", "description", "text", "content", "race", "type"}
        return any(field in obj for field in content_fields)

    # ----------------------------------------------------------------------
    # Convert a record to a searchable text blob
    # ----------------------------------------------------------------------
    def _record_to_text(self, record: Dict) -> str:
        """
        Build a searchable text from the record.
        If the record has a 'content' field, use it (optionally prefixed with its type).
        Otherwise, fall back to the full JSON representation.
        """
        # Prefer 'content' field
        content = record.get('content')
        if content:
            # Include type if available
            type_ = record.get('type')
            if type_:
                return f"Type: {type_}\nContent: {content}"
            return content

        # If no content field, fall back to full JSON dump (lossless)
        return json.dumps(record, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------------
    # Build metadata for the record
    # ----------------------------------------------------------------------
    def _record_to_metadata(self, record: Dict, source_file: Path) -> Dict[str, Any]:
        """
        Extract metadata fields from the record and add file information.
        """
        metadata = {
            # File/adventure info
            "source_file": str(source_file.relative_to(self.raw_root.parent.parent)
                               if source_file.is_absolute() else source_file),
            "adventure": self.adventure_name,
            "added_at": datetime.now().isoformat(),
        }

        # Extract title and type from record if they exist
        if "title" in record and isinstance(record["title"], (str, int, float, bool)):
            metadata["title"] = record["title"]
        if "type" in record and isinstance(record["type"], (str, int, float, bool)):
            metadata["type"] = record["type"]

        # Optionally include other simple fields (strings, numbers, booleans)
        for key, value in record.items():
            if key in metadata:  # skip already added
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = value
            # For nested structures, optionally store a summary or skip
            elif isinstance(value, (dict, list)):
                # Keep as is if small, else indicate truncation
                as_str = json.dumps(value, ensure_ascii=False)
                if len(as_str) < 500:
                    metadata[key] = value
                else:
                    metadata[key] = f"[{key} data - truncated]"
            else:
                metadata[key] = str(value)[:500]

        return metadata

    # ----------------------------------------------------------------------
    # Batch insertion into Faiss
    # ----------------------------------------------------------------------
    def _insert_batch(self, texts: List[str], metadatas: List[Dict]) -> int:
        """
        Insert a batch of records into the vector database.
        Returns the number of successfully inserted records.
        """
        if not texts:
            return 0
        try:
            inserted_ids = self.db.insert(texts, metadatas)
            if self.debug:
                sample = inserted_ids[:3]
                sample_str = ", ".join(map(str, sample))
                if len(inserted_ids) > 3:
                    sample_str += f", … (total {len(inserted_ids)})"
                print(f"   ✅ DEBUG: Inserted batch of {len(inserted_ids)} records (IDs: {sample_str})")
            return len(inserted_ids)
        except Exception as e:
            print(f"   ❌ Batch insert failed ({len(texts)} records): {e}")
            if self.debug:
                print("      ↪  DEBUG: Falling back to single inserts...")
            successful = 0
            for i, (t, m) in enumerate(zip(texts, metadatas)):
                try:
                    doc_id = self.db.insert_single(t, m)
                    successful += 1
                    if self.debug and i < 3:  # show first few successful IDs
                        print(f"         ✅ Inserted single ID: {doc_id}")
                except Exception as e2:
                    if self.debug:
                        print(f"         ❌ Failed single insert: {e2}")
            if self.debug:
                print(f"      ↪  Fallback complete: {successful}/{len(texts)} inserted")
            return successful


def main():
    """Command‑line interface with optional debug flag."""
    import argparse
    parser = argparse.ArgumentParser(description="Load all JSON data into Faiss for an adventure.")
    parser.add_argument("adventure", nargs="?", help="Name of the adventure folder")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    adventure_name = args.adventure
    if not adventure_name:
        adventure_name = input("Enter adventure name: ").strip()
        args.debug = True
        if not adventure_name:
            print("No adventure name provided.")
            return

    print(f"🚀 Loading adventure: '{adventure_name}' (debug={args.debug})")
    loader = UniversalAdventureLoader(adventure_name, debug=args.debug)
    loader.run()


if __name__ == "__main__":
    main()