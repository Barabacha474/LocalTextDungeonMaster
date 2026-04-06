from typing import List, Dict, Optional
from Codes.Databases.AdventureLogger import AdventureLogger
from Codes.Databases.FaissVectorDB import FAISSVectorDB


class AdventureContext:
    """
    Central context manager for the adventure.

    Responsibilities:
    - SQL history (turns)
    - Vector memory (RAG)
    - Player input state
    - Undo / rollback
    - Memory lifecycle

    IMPORTANT:
    - Does NOT build prompts
    - Does NOT know about engine or config
    """

    def __init__(self, sql_db: AdventureLogger, vector_db: FAISSVectorDB):
        self.sql_db = sql_db
        self.vector_db = vector_db

        self._player_input: Optional[str] = None

    # =========================================================
    # PLAYER INPUT
    # =========================================================

    def set_player_input(self, text: str):
        self._player_input = text

    def get_player_input(self) -> Optional[str]:
        return self._player_input

    # =========================================================
    # SQL CONTEXT (LOGS)
    # =========================================================

    def get_last_turns(
        self,
        n: int,
        roles: Optional[List[str]] = None
    ) -> List[Dict]:
        turns = self.sql_db.get_last_n_turns(n)

        if roles:
            turns = [t for t in turns if t["role"] in roles]

        return turns

    def get_turns_range(
        self,
        start: int,
        end: int,
        roles: Optional[List[str]] = None
    ) -> List[Dict]:
        turns = self.sql_db.get_turns_range(start, end)

        if roles:
            turns = [t for t in turns if t["role"] in roles]

        return turns

    def get_turn_count(self) -> int:
        return self.sql_db.get_turn_count()

    def get_latest_turn(self) -> Optional[Dict]:
        return self.sql_db.get_latest_turn()

    # =========================================================
    # LOGGING
    # =========================================================

    def log_turn(
        self,
        turn_id: int,
        role: str,
        content: str,
        model_name: Optional[str] = None,
        seed: Optional[int] = None
    ):
        return self.sql_db.write(
            turn_id=turn_id,
            role=role,
            content=content,
            model_name=model_name,
            seed=seed
        )

    # =========================================================
    # DELETE / ROLLBACK
    # =========================================================

    def delete_last_turn(self) -> bool:
        last = self.sql_db.get_latest_turn()
        if not last:
            return False
        return self.sql_db.delete(last["turn_id"])

    def delete_turn(self, turn_id: int) -> bool:
        return self.sql_db.delete(turn_id)

    def delete_turns_range(self, start: int, end: int):
        for t in range(start, end + 1):
            self.sql_db.delete(t)

    def clear_all_turns(self):
        self.sql_db.clear_all_turns()

    def delete_turn_roles(self, turn_id: int, roles: List[str]) -> int:
        """
        Delete only specific roles within a turn.
        """
        turns = self.sql_db.get_turns_range(turn_id, turn_id)

        to_delete_sql_ids = [
            t["sql_id"]
            for t in turns
            if t["role"] in roles
        ]

        for sql_id in to_delete_sql_ids:
            self.sql_db.delete_by_sql_id(sql_id)

        return len(to_delete_sql_ids)

    # =========================================================
    # VECTOR MEMORY (RAG)
    # =========================================================

    def get_relevant_info(
        self,
        query: str,
        k_per_cascade: int = 5,
        number_of_cascades: int = 1,
        threshold: float = 0.3,
        chunk_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Vector search over memory.

        Fully configurable from outside.
        """
        return self.vector_db.search(
            query=query,
            k_per_cascade=k_per_cascade,
            number_of_cascades=number_of_cascades,
            threshold=threshold,
            chunk_size=chunk_size
        )

    def get_all_memory(self) -> List[Dict]:
        return self.vector_db.get_all_documents()

    # =========================================================
    # GLOBAL SUMMARY (NEW)
    # =========================================================

    def get_latest_global_summary(self) -> Optional[str]:
        """
        Returns last global summary from memory.
        Used by narrator and planner.
        """
        docs = self.vector_db.get_all_documents()

        summaries = [
            d["text"]
            for d in docs
            if d.get("metadata", {}).get("type") == "global_summary"
        ]

        if summaries:
            return summaries[-1]

        return None

    # =========================================================
    # SEARCH QUERY BUILDER (NEW)
    # =========================================================

    def build_search_query(
        self,
        user_input: str,
        n_history: int = 5
    ) -> str:
        """
        Build context-aware RAG query.

        Combines:
        - recent turns
        - current player input
        """

        history = self.get_last_turns(n_history)

        history_text = "\n".join(
            f"{t['role']}: {t['content']}"
            for t in history
        )

        return f"{history_text}\n\nCurrent query: {user_input}"

    # =========================================================
    # MEMORY MANAGEMENT
    # =========================================================

    def save_memory(
        self,
        text: str,
        turn_start: int,
        turn_end: int
    ) -> int:
        label = f"Memory of turns {turn_start}-{turn_end}"

        metadata = {
            "type": "memory",
            "turn_start": turn_start,
            "turn_end": turn_end,
            "label": label
        }

        print(f"\n MEMORY SAVED: {text} \n")

        return self.vector_db.insert_single(text=text, metadata=metadata)

    def delete_memory_by_turn_range(self, turn_start: int, turn_end: int):
        docs = self.vector_db.get_all_documents()

        to_delete = [
            d["id"]
            for d in docs
            if d.get("metadata", {}).get("type") == "memory"
            and d["metadata"].get("turn_start") == turn_start
            and d["metadata"].get("turn_end") == turn_end
        ]

        if to_delete:
            self.vector_db.delete(to_delete)
            self.vector_db.save()

    def delete_memories_after_turn(self, turn_id: int):
        docs = self.vector_db.get_all_documents()

        to_delete = [
            d["id"]
            for d in docs
            if d.get("metadata", {}).get("type") == "memory"
            and d["metadata"].get("turn_end", 0) > turn_id
        ]

        if to_delete:
            self.vector_db.delete(to_delete)
            self.vector_db.save()

    # =========================================================
    # HIGH-LEVEL OPERATIONS
    # =========================================================

    def undo_last_turn(self):
        last = self.get_latest_turn()
        if not last:
            return False

        turn_id = last["turn_id"]

        success = self.delete_turn(turn_id)

        if success:
            self.delete_memories_after_turn(turn_id - 1)

        return success

    def regenerate_last_turn(self, roles: Optional[List[str]] = None):
        last_turns = self.get_last_turns(1)

        if not last_turns:
            return None

        turn_id = last_turns[0]["turn_id"]

        roles = roles or ["Narrator"]

        deleted = self.delete_turn_roles(turn_id, roles)

        return {
            "turn_id": turn_id,
            "deleted_count": deleted
        }