import pytest
from Codes.Databases.AdventureLogger import AdventureLogger


# =========================
# FIXTURE
# =========================

@pytest.fixture
def logger(tmp_path):
    """
    Creates a fresh isolated database for each test case.
    Ensures no side effects between tests.
    """
    log = AdventureLogger(adventure_name="test_db", storage_path=str(tmp_path))
    yield log
    log.close()


# =========================
# WRITE / READ
# =========================

def test_write_and_read_by_turn_id(logger):
    """Verify that writing and reading by turn_id works correctly."""
    logger.write(1, "Player", "Hello")

    result = logger.read_by_turn_id(1)

    assert result is not None
    assert result["content"] == "Hello"
    assert result["role"] == "Player"


def test_read_nonexistent_turn(logger):
    """Reading a non-existent turn_id should return None."""
    assert logger.read_by_turn_id(999) is None


def test_read_by_sql_id(logger):
    """Verify reading by immutable sql_id."""
    logger.write(1, "Narrator", "Test")

    turn = logger.read_by_turn_id(1)
    sql_id = turn["sql_id"]

    result = logger.read_by_sql_id(sql_id)

    assert result["content"] == "Test"


def test_read_alias(logger):
    """Alias read() should behave exactly like read_by_turn_id()."""
    logger.write(1, "A", "text")

    assert logger.read(1)["content"] == "text"


# =========================
# UPDATE
# =========================

def test_update_by_turn_id_success(logger):
    """Ensure updating an existing turn by turn_id succeeds."""
    logger.write(1, "Player", "Old")

    success = logger.update_by_turn_id(1, "New")

    assert success is True
    assert logger.read_by_turn_id(1)["content"] == "New"


def test_update_by_turn_id_fail(logger):
    """Updating a non-existent turn_id should return False."""
    assert logger.update_by_turn_id(999, "Fail") is False


def test_update_with_seed(logger):
    """Verify that updating content together with seed works correctly."""
    logger.write(1, "Player", "Text", seed=42)

    logger.update_by_turn_id(1, "Updated", seed=99)

    result = logger.read_by_turn_id(1)
    assert result["seed"] == 99


def test_update_by_sql_id(logger):
    """Ensure updating by sql_id works correctly."""
    logger.write(1, "A", "old")
    sql_id = logger.read_by_turn_id(1)["sql_id"]

    success = logger.update_by_sql_id(sql_id, "new")

    assert success is True
    assert logger.read_by_turn_id(1)["content"] == "new"


# =========================
# DELETE + REORDER
# =========================

def test_delete_by_turn_id_and_reorder(logger):
    """
    Validate critical behavior:
    after deletion, subsequent turn_ids must be renumbered.
    """
    logger.write(1, "A", "1")
    logger.write(2, "A", "2")
    logger.write(3, "A", "3")

    logger.delete_by_turn_id(2)

    # Former turn 3 should now become turn 2
    turn = logger.read_by_turn_id(2)

    assert turn["content"] == "3"


def test_delete_nonexistent_turn(logger):
    """Deleting a non-existent turn_id should return False."""
    assert logger.delete_by_turn_id(999) is False


def test_delete_by_sql_id(logger):
    """Verify deletion by sql_id and correct renumbering."""
    logger.write(1, "A", "1")
    logger.write(2, "A", "2")

    sql_id = logger.read_by_turn_id(1)["sql_id"]

    logger.delete_by_sql_id(sql_id)

    # Former turn 2 should now become turn 1
    turn = logger.read_by_turn_id(1)

    assert turn["content"] == "2"


def test_delete_alias(logger):
    """Alias delete() should behave like delete_by_turn_id()."""
    logger.write(1, "A", "text")

    assert logger.delete(1) is True
    assert logger.read(1) is None


# =========================
# GETTERS
# =========================

def test_get_turn_count(logger):
    """Ensure turn count reflects the highest turn_id correctly."""
    assert logger.get_turn_count() == 0

    logger.write(1, "A", "1")
    logger.write(2, "A", "2")

    assert logger.get_turn_count() == 2


def test_get_latest_turn(logger):
    """Verify retrieval of the most recent turn."""
    logger.write(1, "A", "1")
    logger.write(2, "A", "2")

    latest = logger.get_latest_turn()

    assert latest["turn_id"] == 2


def test_get_latest_turn_empty(logger):
    """If database is empty, latest turn should be None."""
    assert logger.get_latest_turn() is None


# =========================
# RANGE / LAST N
# =========================

def test_get_last_n_turns(logger):
    """Verify retrieval of the last N turns."""
    for i in range(1, 6):
        logger.write(i, "A", str(i))

    result = logger.get_last_n_turns(3)

    assert len(result) == 3
    assert result[0]["turn_id"] == 3
    assert result[-1]["turn_id"] == 5


def test_get_last_n_empty(logger):
    """Empty database should return an empty list."""
    assert logger.get_last_n_turns(5) == []


def test_get_turns_range(logger):
    """Verify retrieval of turns within a specified range."""
    for i in range(1, 6):
        logger.write(i, "A", str(i))

    result = logger.get_turns_range(2, 4)

    assert len(result) == 3
    assert result[0]["turn_id"] == 2
    assert result[-1]["turn_id"] == 4


def test_get_turns_range_out_of_bounds(logger):
    """Out-of-range queries should return an empty list."""
    logger.write(1, "A", "1")

    result = logger.get_turns_range(5, 10)

    assert result == []


# =========================
# SEARCH
# =========================

def test_search_content_found(logger):
    """Ensure keyword search returns matching entries."""
    logger.write(1, "A", "dragon attack")
    logger.write(2, "A", "peaceful village")

    result = logger.search_content("dragon")

    assert len(result) == 1
    assert "dragon" in result[0]["content"]


def test_search_content_not_found(logger):
    """Search with no matches should return an empty list."""
    logger.write(1, "A", "hello")

    assert logger.search_content("xyz") == []


def test_search_limit(logger):
    """Ensure search respects result limit."""
    for i in range(10):
        logger.write(i + 1, "A", "same keyword")

    result = logger.search_content("keyword", limit=5)

    assert len(result) == 5


# =========================
# CLEAR / DELETE DB
# =========================

def test_clear_all_turns(logger):
    """Verify that all records are removed while schema remains."""
    logger.write(1, "A", "1")
    logger.clear_all_turns()

    assert logger.get_turn_count() == 0


def test_delete_database(tmp_path):
    """
    Ensure the database file is physically removed from disk.
    """
    logger = AdventureLogger("test_db", str(tmp_path))
    path = logger.db_path

    assert path.exists()

    logger.delete_database()

    assert not path.exists()


# =========================
# CONTEXT MANAGER
# =========================

def test_context_manager(tmp_path):
    """Ensure context manager properly closes the connection."""
    with AdventureLogger("test_db", str(tmp_path)) as logger:
        logger.write(1, "A", "test")

    # If no exception occurred, __exit__ worked correctly


# =========================
# CLASS METHODS
# =========================

def test_list_adventures(tmp_path):
    """Ensure list_adventures returns existing database names."""
    AdventureLogger("a1", str(tmp_path))
    AdventureLogger("a2", str(tmp_path))

    result = AdventureLogger.list_adventures(str(tmp_path))

    assert set(result) == {"a1", "a2"}