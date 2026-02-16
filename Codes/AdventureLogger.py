import sqlite3
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class AdventureLogger:
    """
    SQLite-based logger for adventure turns.
    Each adventure gets its own database file.
    Now with:
      - sql_id: immutable autoincrement primary key
      - turn_id: logical turn number (contiguous, renumbered on delete)
      - seed: integer seed used for generation
      - roles: Narrator, Adventure planner, Player
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
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        safe = "_".join(filter(None, safe.split("_")))
        return safe

    def _initialize_database(self):
        """Initialize the database with turns table if it doesn't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create turns table with both sql_id (immutable) and turn_id (logical)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                sql_id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_name TEXT,
                seed INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for fast retrieval
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn_id ON turns(turn_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_sql_id ON turns(sql_id)")

        self.conn.commit()

    def _reorder_turn_ids_after_delete(self, deleted_turn_id: int):
        """
        After deleting a turn, decrement all turn_ids that were greater than it,
        so they remain contiguous.
        """
        self.cursor.execute(
            "UPDATE turns SET turn_id = turn_id - 1 WHERE turn_id > ?",
            (deleted_turn_id,)
        )
        self.conn.commit()

    def write(self, turn_id: int, role: str, content: str, model_name: Optional[str] = None, seed: Optional[int] = None) -> int:
        """
        Write a new turn to the database.

        Args:
            turn_id: id of internal adventure turn
            role: One of 'Narrator', 'Adventure planner', 'Player' (or custom)
            content: Text content of the turn
            model_name: name of the model used for generation
            seed: Optional integer seed used for generation

        Returns:
            int: The new logical turn_id (sequential)
        """
        # Validate role if desired (optional)
        # if role not in ['Narrator', 'Adventure planner', 'Player', 'DM']:
        #     raise ValueError(f"Unknown role: {role}")

        self.cursor.execute(
            "INSERT INTO turns (turn_id, role, content, model_name, seed) VALUES (?, ?, ?, ?, ?)",
            (turn_id, role, content, model_name, seed)
        )
        self.conn.commit()

        return turn_id

    def read_by_turn_id(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """
        Read a specific turn by its logical turn_id.

        Args:
            turn_id: logical turn number

        Returns:
            dict with turn data or None if not found
        """
        self.cursor.execute(
            "SELECT sql_id, turn_id, role, content, seed, timestamp FROM turns WHERE turn_id = ?",
            (turn_id,)
        )
        row = self.cursor.fetchone()
        if row:
            return {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
            }
        return None

    def read_by_sql_id(self, sql_id: int) -> Optional[Dict[str, Any]]:
        """
        Read a specific turn by its immutable sql_id.

        Args:
            sql_id: autoincrement primary key

        Returns:
            dict with turn data or None if not found
        """
        self.cursor.execute(
            "SELECT sql_id, turn_id, role, content, seed, timestamp FROM turns WHERE sql_id = ?",
            (sql_id,)
        )
        row = self.cursor.fetchone()
        if row:
            return {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
            }
        return None

    # Alias for backward compatibility
    def read(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Alias for read_by_turn_id."""
        return self.read_by_turn_id(turn_id)

    def update_by_turn_id(self, turn_id: int, content: str, seed: Optional[int] = None) -> bool:
        """
        Update the content and optionally seed of a specific turn (by turn_id).

        Args:
            turn_id: logical turn number
            content: New content
            seed: Optional new seed (if None, seed is not updated)

        Returns:
            bool: True if update successful, False if turn not found
        """
        if seed is not None:
            self.cursor.execute(
                "UPDATE turns SET content = ?, seed = ? WHERE turn_id = ?",
                (content, seed, turn_id)
            )
        else:
            self.cursor.execute(
                "UPDATE turns SET content = ? WHERE turn_id = ?",
                (content, turn_id)
            )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def update_by_sql_id(self, sql_id: int, content: str, seed: Optional[int] = None) -> bool:
        """
        Update a turn by its immutable sql_id.
        """
        if seed is not None:
            self.cursor.execute(
                "UPDATE turns SET content = ?, seed = ? WHERE sql_id = ?",
                (content, seed, sql_id)
            )
        else:
            self.cursor.execute(
                "UPDATE turns SET content = ? WHERE sql_id = ?",
                (content, sql_id)
            )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_by_turn_id(self, turn_id: int) -> bool:
        """
        Delete a specific turn by its logical turn_id and renumber later turns.

        Args:
            turn_id: logical turn number to delete

        Returns:
            bool: True if deletion successful, False if turn not found
        """
        # Check existence
        self.cursor.execute("SELECT COUNT(*) FROM turns WHERE turn_id = ?", (turn_id,))
        if self.cursor.fetchone()[0] == 0:
            return False

        total_before = self.get_turn_count()

        # Delete the turn
        self.cursor.execute("DELETE FROM turns WHERE turn_id = ?", (turn_id,))

        # Renumber later turns if we didn't delete the last one
        if turn_id != total_before:
            self._reorder_turn_ids_after_delete(turn_id)

        self.conn.commit()
        return True

    def delete_by_sql_id(self, sql_id: int) -> bool:
        """
        Delete a turn by its immutable sql_id, then renumber logical turn_ids.

        Args:
            sql_id: immutable primary key

        Returns:
            bool: True if deletion successful, False if turn not found
        """
        # First get the turn_id of the row to delete
        self.cursor.execute("SELECT turn_id FROM turns WHERE sql_id = ?", (sql_id,))
        row = self.cursor.fetchone()
        if not row:
            return False
        turn_id = row[0]

        # Now delete by sql_id
        self.cursor.execute("DELETE FROM turns WHERE sql_id = ?", (sql_id,))

        # Renumber later turns
        total_before = self.get_turn_count() + 1  # before deletion count
        if turn_id != total_before:
            self._reorder_turn_ids_after_delete(turn_id)

        self.conn.commit()
        return True

    # Keep old delete method for compatibility (uses turn_id)
    def delete(self, turn_id: int) -> bool:
        """Alias for delete_by_turn_id."""
        return self.delete_by_turn_id(turn_id)

    def get_last_n_turns(self, n: int) -> List[Dict[str, Any]]:
        """
        Get all documents for the last N distinct turn IDs from the database,
        ordered by turn_id ascending.

        Args:
            n: Number of distinct turn IDs to retrieve

        Returns:
            List of turn dictionaries, each containing both ids.
        """
        # Get the maximum turn_id
        max_turn = self.get_turn_count()
        if max_turn == 0:
            return []

        start_turn = max(1, max_turn - (n - 1))

        query = """
            SELECT sql_id, turn_id, role, content, seed, timestamp 
            FROM turns 
            WHERE turn_id >= ? AND turn_id <= ?
            ORDER BY turn_id ASC
        """
        self.cursor.execute(query, (start_turn, max_turn))
        rows = self.cursor.fetchall()

        return [
            {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
            }
            for row in rows
        ]

    def get_turns_range(self, start_turn_id: int, end_turn_id: int) -> List[Dict[str, Any]]:
        """
        Get turns within a specific turn_id range (inclusive).

        Args:
            start_turn_id: Starting logical turn ID
            end_turn_id: Ending logical turn ID

        Returns:
            List of turn dictionaries
        """
        self.cursor.execute(
            """
            SELECT sql_id, turn_id, role, content, seed, timestamp 
            FROM turns 
            WHERE turn_id >= ? AND turn_id <= ?
            ORDER BY turn_id ASC
            """,
            (start_turn_id, end_turn_id)
        )
        rows = self.cursor.fetchall()
        return [
            {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
            }
            for row in rows
        ]

    def get_turn_count(self) -> int:
        """Get the highest turn number (max turn_id) in the database."""
        self.cursor.execute("SELECT MAX(turn_id) FROM turns")
        max_turn = self.cursor.fetchone()[0]
        return max_turn if max_turn is not None else 0

    def get_latest_turn(self) -> Optional[Dict[str, Any]]:
        """Get the most recent turn (highest turn_id)."""
        self.cursor.execute(
            "SELECT sql_id, turn_id, role, content, seed, timestamp FROM turns ORDER BY turn_id DESC LIMIT 1"
        )
        row = self.cursor.fetchone()
        if row:
            return {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
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
            SELECT sql_id, turn_id, role, content, seed, timestamp 
            FROM turns 
            WHERE content LIKE ? 
            ORDER BY turn_id DESC 
            LIMIT ?
            """,
            (f"%{keyword}%", limit)
        )
        rows = self.cursor.fetchall()
        return [
            {
                'sql_id': row[0],
                'turn_id': row[1],
                'role': row[2],
                'content': row[3],
                'seed': row[4],
                'timestamp': row[5]
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @classmethod
    def list_adventures(cls, storage_path: str = "../adventure_logs") -> List[str]:
        path = Path(storage_path)
        if not path.exists():
            return []
        return [file.stem for file in path.glob("*.db")]