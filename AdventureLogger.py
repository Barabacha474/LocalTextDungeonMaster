import sqlite3
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class AdventureLogger:
    """
    SQLite-based logger for adventure turns.
    Each adventure gets its own database file.
    """

    def __init__(self, adventure_name: str = "vanilla_fantasy", storage_path: str = "./adventure_logs"):
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

        # Create turns table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('System', 'Player')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster retrieval by id
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn_id ON turns(id)")

        self.conn.commit()

    def write(self, role: str, content: str) -> int:
        """
        Write a new turn to the database.

        Args:
            role: 'System' or 'Player'
            content: Text content of the turn

        Returns:
            int: The ID of the inserted turn
        """
        if role not in ['System', 'Player']:
            raise ValueError("Role must be 'System' or 'Player'")

        self.cursor.execute(
            "INSERT INTO turns (role, content) VALUES (?, ?)",
            (role, content)
        )
        self.conn.commit()

        return self.cursor.lastrowid

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
        Delete a specific turn.

        Args:
            turn_id: ID of the turn to delete

        Returns:
            bool: True if deletion successful, False if turn not found
        """
        self.cursor.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
        self.conn.commit()

        return self.cursor.rowcount > 0

    def get_last_n_turns(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the last N turns from the database.

        Args:
            n: Number of turns to retrieve
            include_system: Whether to include System turns

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
        self.cursor.execute("DELETE FROM sqlite_sequence WHERE name='turns'")
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
    def list_adventures(cls, storage_path: str = "./adventure_logs") -> List[str]:
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


# Example usage
if __name__ == "__main__":
    # Create/connect to an adventure log
    logger = AdventureLogger("The Lost Temple")

    # Write some turns
    turn1_id = logger.write("System", "You stand before an ancient temple covered in vines.")
    turn2_id = logger.write("Player", "I carefully examine the entrance for traps.")
    turn3_id = logger.write("System", "You notice a pressure plate hidden under moss.")

    print(f"Turn 1 ID: {turn1_id}")
    print(f"Turn 2 ID: {turn2_id}")
    print(f"Turn 3 ID: {turn3_id}")

    # Read a specific turn
    turn = logger.read(turn2_id)
    print(f"\nTurn {turn2_id}: {turn}")

    # Update a turn
    logger.update(turn2_id, "I carefully examine the entrance for traps and markings.")

    # Get last 5 turns
    last_turns = logger.get_last_n_turns(5)
    print(f"\nLast {len(last_turns)} turns:")
    for t in last_turns:
        print(f"  [{t['id']}] {t['role']}: {t['content'][:50]}...")

    # Get turn count
    count = logger.get_turn_count()
    print(f"\nTotal turns: {count}")

    # Search for content
    results = logger.search_content("temple")
    print(f"\nTurns mentioning 'temple': {len(results)}")

    # List all adventures
    adventures = AdventureLogger.list_adventures("./adventure_logs")
    print(f"\nAvailable adventures: {adventures}")

    # Close connection
    logger.close()