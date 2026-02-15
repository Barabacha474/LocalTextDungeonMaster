import sqlite3
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class AdventureLogger:
    """
    SQLite-based logger for adventure turns.
    Each adventure gets its own database file.
    Now with sequential IDs that always match document position.
    """

    def __init__(self, adventure_name: str = "vanilla_fantasy", storage_path: str = "../adventure_logs"):
        """
        Initialize or connect to an adventure database.

        Args:
            adventure_name: Name of the adventure (used for database filename)
            storage_path: Directory where database files are stored
        """
        self.adventure_name = adventure_name
        self.storage_path = Path(storage_path)

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Sanitize adventure name for filename
        safe_name = self._sanitize_filename(adventure_name)
        self.db_path = self.storage_path / f"{safe_name}.db"

        # Initialize database connection
        self.conn = None
        self.cursor = None
        self._initialize_database()

    def _sanitize_filename(self, name: str) -> str:
        """Convert adventure name to safe filename."""
        # Replace unsafe characters with underscores
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        # Remove multiple underscores
        safe = "_".join(filter(None, safe.split("_")))
        return safe

    def _initialize_database(self):
        """Initialize the database with turns table if it doesn't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create turns table without autoincrement
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL CHECK(role IN ('DM', 'Player')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster retrieval by id
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn_id ON turns(id)")

        self.conn.commit()

    def _get_next_id(self) -> int:
        """Calculate the next ID based on current document count."""
        self.cursor.execute("SELECT COUNT(*) FROM turns")
        count = self.cursor.fetchone()[0]
        return count + 1

    def _reorder_ids_after_delete(self, deleted_id: int):
        """
        Reorder IDs after deletion so they remain sequential.
        This decrements IDs of all documents with ID > deleted_id.
        """
        # SQLite UPDATE doesn't support ORDER BY, but we don't need it
        # We just need to decrement IDs of all rows with higher IDs
        self.cursor.execute(
            "UPDATE turns SET id = id - 1 WHERE id > ?",
            (deleted_id,)
        )
        self.conn.commit()

    def write(self, role: str, content: str) -> int:
        """
        Write a new turn to the database.

        Args:
            role: 'DM' or 'Player'
            content: Text content of the turn

        Returns:
            int: The ID of the inserted turn
        """
        if role not in ['DM', 'Player']:
            raise ValueError("Role must be 'DM' or 'Player'")

        # Calculate next ID (count + 1)
        new_id = self._get_next_id()

        self.cursor.execute(
            "INSERT INTO turns (id, role, content) VALUES (?, ?, ?)",
            (new_id, role, content)
        )
        self.conn.commit()

        return new_id

    def read(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """
        Read a specific turn by ID.

        Args:
            turn_id: ID of the turn to read

        Returns:
            dict with turn data or None if not found
        """
        self.cursor.execute(
            "SELECT id, role, content, timestamp FROM turns WHERE id = ?",
            (turn_id,)
        )

        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'timestamp': row[3]
            }
        return None

    def update(self, turn_id: int, content: str) -> bool:
        """
        Update the content of a specific turn.

        Args:
            turn_id: ID of the turn to update
            content: New content

        Returns:
            bool: True if update successful, False if turn not found
        """
        self.cursor.execute(
            "UPDATE turns SET content = ? WHERE id = ?",
            (content, turn_id)
        )
        self.conn.commit()

        return self.cursor.rowcount > 0

    def delete(self, turn_id: int) -> bool:
        """
        Delete a specific turn and reorder remaining IDs.

        Args:
            turn_id: ID of the turn to delete

        Returns:
            bool: True if deletion successful, False if turn not found
        """
        # First check if the turn exists
        self.cursor.execute("SELECT COUNT(*) FROM turns WHERE id = ?", (turn_id,))
        if self.cursor.fetchone()[0] == 0:
            return False

        # Get the total count before deletion
        total_before = self.get_turn_count()

        # Delete the turn
        self.cursor.execute("DELETE FROM turns WHERE id = ?", (turn_id,))

        # Check if we're deleting the last turn (no need to reorder)
        if turn_id != total_before:
            # Reorder IDs of turns that come after
            self._reorder_ids_after_delete(turn_id)

        self.conn.commit()

        return True

    def get_last_n_turns(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the last N turns from the database.

        Args:
            n: Number of turns to retrieve

        Returns:
            List of turn dictionaries, ordered by turn ID (ascending)
        """
        query = """
            SELECT id, role, content, timestamp 
            FROM turns 
            ORDER BY id DESC 
            LIMIT ?
        """

        self.cursor.execute(query, (n,))
        rows = self.cursor.fetchall()

        # Convert to list of dictionaries
        turns = []
        for row in reversed(rows):  # Reverse to get chronological order
            turns.append({
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'timestamp': row[3]
            })

        return turns

    def get_turns_range(self, start_id: int, end_id: int) -> List[Dict[str, Any]]:
        """
        Get turns within a specific ID range (inclusive).

        Args:
            start_id: Starting turn ID
            end_id: Ending turn ID

        Returns:
            List of turn dictionaries
        """
        self.cursor.execute(
            """
            SELECT id, role, content, timestamp 
            FROM turns 
            WHERE id >= ? AND id <= ?
            ORDER BY id ASC
            """,
            (start_id, end_id)
        )

        rows = self.cursor.fetchall()
        return [
            {
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'timestamp': row[3]
            }
            for row in rows
        ]

    def get_turn_count(self) -> int:
        """Get total number of turns in the database."""
        self.cursor.execute("SELECT COUNT(*) FROM turns")
        return self.cursor.fetchone()[0]

    def get_latest_turn(self) -> Optional[Dict[str, Any]]:
        """Get the most recent turn."""
        self.cursor.execute(
            "SELECT id, role, content, timestamp FROM turns ORDER BY id DESC LIMIT 1"
        )
        row = self.cursor.fetchone()

        if row:
            return {
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'timestamp': row[3]
            }
        return None

    def search_content(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for turns containing a keyword in their content.

        Args:
            keyword: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching turns
        """
        self.cursor.execute(
            """
            SELECT id, role, content, timestamp 
            FROM turns 
            WHERE content LIKE ? 
            ORDER BY id DESC 
            LIMIT ?
            """,
            (f"%{keyword}%", limit)
        )

        rows = self.cursor.fetchall()
        return [
            {
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'timestamp': row[3]
            }
            for row in rows
        ]

    def clear_all_turns(self):
        """Clear all turns from the database (keep the structure)."""
        self.cursor.execute("DELETE FROM turns")
        self.conn.commit()

    def delete_database(self):
        """Delete the entire database file."""
        self.close()

        if self.db_path.exists():
            os.remove(self.db_path)
            return True
        return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()

    def __del__(self):
        """Destructor - ensure connection is closed."""
        self.close()

    @classmethod
    def list_adventures(cls, storage_path: str = "../adventure_logs") -> List[str]:
        """
        List all adventure databases in the storage directory.

        Args:
            storage_path: Directory to search

        Returns:
            List of adventure names (without .db extension)
        """
        path = Path(storage_path)
        if not path.exists():
            return []

        adventures = []
        for file in path.glob("*.db"):
            adventures.append(file.stem)

        return adventures


# Example usage demonstrating the new ID behavior
if __name__ == "__main__":
    # Create/connect to an adventure log
    logger = AdventureLogger("Sequential Test")

    # Clear any existing data
    logger.clear_all_turns()

    # Write some turns
    print("=== Initial insertions ===")
    turn1_id = logger.write("DM", "First message")
    turn2_id = logger.write("Player", "Second message")
    turn3_id = logger.write("DM", "Third message")
    turn4_id = logger.write("Player", "Fourth message")

    print(f"Turn IDs: {turn1_id}, {turn2_id}, {turn3_id}, {turn4_id}")

    # Show all turns
    all_turns = logger.get_turns_range(1, logger.get_turn_count())
    for turn in all_turns:
        print(f"ID {turn['id']}: {turn['content']}")

    # Delete turn with ID 2
    print("\n=== Deleting turn with ID 2 ===")
    logger.delete(2)

    # Show remaining turns (IDs should be 1, 2, 3)
    all_turns = logger.get_turns_range(1, logger.get_turn_count())
    for turn in all_turns:
        print(f"ID {turn['id']}: {turn['content']}")

    # Add a new turn - should get ID 4 (3 existing + 1)
    print("\n=== Adding new turn after deletion ===")
    new_turn_id = logger.write("DM", "Fifth message (new)")
    print(f"New turn ID: {new_turn_id}")

    # Show all turns (IDs should be 1, 2, 3, 4)
    all_turns = logger.get_turns_range(1, logger.get_turn_count())
    for turn in all_turns:
        print(f"ID {turn['id']}: {turn['content']}")

    # Close connection
    logger.close()