import pytest
import numpy as np
from Codes.Databases.FaissVectorDB import FAISSVectorDB


# =========================
# MOCK EMBEDDING
# =========================

class DummyEmbedding:
    """
    Deterministic embedding model based on character codes.
    Produces distinguishable vectors for different texts.
    """
    def encode(self, texts):
        vectors = []
        for t in texts:
            vec = np.zeros(384, dtype="float32")
            for i, c in enumerate(t[:384]):
                vec[i] = ord(c) % 100  # simple deterministic signal
            vectors.append(vec)
        return np.array(vectors)


# =========================
# FIXTURE
# =========================

@pytest.fixture(autouse=True)
def mock_embedding(monkeypatch):
    """Replace embedding model with deterministic dummy."""
    def mock_model(self):
        return DummyEmbedding()

    monkeypatch.setattr(FAISSVectorDB, "_load_embedding_model", mock_model)


@pytest.fixture
def db(tmp_path):
    """
    Creates isolated DB with mocked embeddings.
    """
    db = FAISSVectorDB("test_adventure", str(tmp_path))
    yield db
    db.delete_adventure()


# =========================
# INSERT
# =========================

def test_insert_and_retrieve(db):
    """Inserted documents should be retrievable with correct content."""
    ids = db.insert(["hello", "world"])

    assert len(ids) == 2
    assert db.get_by_id(ids[0])["text"] == "hello"
    assert db.get_by_id(ids[1])["text"] == "world"


def test_insert_empty(db):
    """Inserting empty list should return empty result."""
    assert db.insert([]) == []


def test_insert_single(db):
    """Single insert should behave like batch insert."""
    doc_id = db.insert_single("test")

    doc = db.get_by_id(doc_id)
    assert doc["text"] == "test"
    assert doc["id"] == doc_id


# =========================
# DELETE
# =========================

def test_delete_documents(db):
    """Deleting documents should remove only specified entries."""
    ids = db.insert(["a", "b"])

    db.delete([ids[0]])

    assert db.get_by_id(ids[0]) is None
    assert db.get_by_id(ids[1]) is not None
    assert db.get_document_count() == 1


def test_delete_empty_list(db):
    """Deleting empty list should succeed and not affect data."""
    db.insert(["a"])
    assert db.delete([]) is True
    assert db.get_document_count() == 1


def test_delete_adventure(db):
    """Deleting adventure should remove directory from filesystem."""
    path = db.adventure_path

    assert path.exists()

    db.delete_adventure()

    assert not path.exists()


# =========================
# SEARCH
# =========================

def test_search_basic(db):
    """Search should return most relevant document first."""
    db.insert(["dragon", "castle", "knight"])

    results = db.search("dragon")

    assert len(results) > 0
    assert results[0]["text"] == "dragon"


def test_search_empty_db(db):
    """Search on empty DB should return empty list."""
    assert db.search("anything") == []


def test_search_with_threshold(db):
    """High threshold should filter out all results."""
    db.insert(["short", "very very long text"])

    results = db.search("tiny", threshold=0.99)

    assert results == []


def test_search_with_chunking(db):
    """Chunked search should still find relevant documents."""
    db.insert(["dragon story", "castle siege"])

    results = db.search("dragon " * 200, chunk_size=10)

    assert any(r["text"] == "dragon story" for r in results)


def test_cascade_search(db):
    """Cascade search should expand result set."""
    db.insert(["dragon", "fire dragon", "ancient fire dragon"])

    results = db.search("dragon", number_of_cascades=2)

    texts = [r["text"] for r in results]

    assert "dragon" in texts
    assert "fire dragon" in texts


def test_search_with_metadata_filter(db):
    """Search should respect metadata filtering."""
    db.insert(["a", "b"], metadata=[{"type": "x"}, {"type": "y"}])

    results = db.search("a", filter_metadata={"type": "x"})

    assert len(results) > 0
    assert all(r["metadata"]["type"] == "x" for r in results)


# =========================
# GETTERS
# =========================

def test_get_by_ids(db):
    """Should retrieve correct subset of documents."""
    ids = db.insert(["a", "b", "c"])

    docs = db.get_by_ids(ids[:2])

    assert len(docs) == 2
    assert docs[0]["text"] == "a"
    assert docs[1]["text"] == "b"


def test_get_all_documents(db):
    """Should return all inserted documents."""
    db.insert(["a", "b"])

    docs = db.get_all_documents()

    assert len(docs) == 2
    assert {d["text"] for d in docs} == {"a", "b"}


def test_get_document_count(db):
    """Document count should reflect actual stored documents."""
    db.insert(["a", "b"])

    assert db.get_document_count() == 2


def test_get_index_size(db):
    """Index size should match number of stored vectors."""
    db.insert(["a", "b"])

    assert db.get_index_size() == 2


# =========================
# CLEAR
# =========================

def test_clear(db):
    """Clearing DB should remove all documents and reset index."""
    db.insert(["a", "b"])

    db.clear()

    assert db.get_document_count() == 0
    assert db.get_index_size() == 0


def test_clear_on_empty(db):
    """Clearing empty DB should still succeed."""
    assert db.clear() is True


# =========================
# DELETE BY CONDITIONS
# =========================

def test_delete_by_metadata(db):
    """Only documents matching metadata should be deleted."""
    ids = db.insert(["a", "b"], metadata=[{"type": "x"}, {"type": "y"}])

    deleted = db.delete_by_metadata({"type": "x"})

    assert deleted == [ids[0]]
    assert db.get_by_id(ids[0]) is None
    assert db.get_by_id(ids[1]) is not None


def test_delete_where(db):
    """Predicate-based deletion should remove correct documents."""
    ids = db.insert(["short", "very long text"])

    deleted = db.delete_where(lambda d: len(d["text"]) < 6)

    assert deleted == [ids[0]]
    assert db.get_by_id(ids[0]) is None
    assert db.get_by_id(ids[1]) is not None


# =========================
# EXPORT / STATS
# =========================

def test_export_documents(db, tmp_path):
    """Exported file should contain correct data."""
    db.insert(["a"])

    path = db.export_documents(tmp_path / "export.json")

    import json
    with open(path) as f:
        data = json.load(f)

    assert data["total_documents"] == 1
    assert data["documents"][0]["text"] == "a"


def test_get_stats(db):
    """Stats should reflect document counts and types."""
    db.insert(["a", "b"], metadata=[{"type": "x"}, {"type": "x"}])

    stats = db.get_stats()

    assert stats["total_documents"] == 2
    assert stats["active_documents"] == 2
    assert stats["document_types"]["x"] == 2


# =========================
# REBUILD
# =========================

def test_rebuild_index(db):
    """Rebuild should preserve data and index size."""
    db.insert(["a", "b"])

    assert db.rebuild_index() is True
    assert db.get_index_size() == 2


def test_delete_triggers_rebuild(db):
    """After deletion and save, index should reflect new state."""
    ids = db.insert(["a", "b"])

    db.delete([ids[0]])
    db.save()

    assert db.get_document_count() == 1
    assert db.get_index_size() == 1


def test_rebuild_empty(db):
    """Rebuilding empty DB should reset index."""
    db.clear()

    assert db.rebuild_index() is True
    assert db.get_index_size() == 0


# =========================
# CONTEXT
# =========================

def test_context_manager(tmp_path, mock_embedding):
    """Context manager should persist data after exit."""
    with FAISSVectorDB("test", str(tmp_path)) as db:
        db.insert(["a"])

    db2 = FAISSVectorDB("test", str(tmp_path))

    assert db2.get_document_count() == 1


# =========================
# LIST ADVENTURES
# =========================

def test_list_adventures(tmp_path, mock_embedding):
    """Should return only valid FAISS databases."""
    db1 = FAISSVectorDB("adv1", str(tmp_path))
    db1.insert(["a"])

    db2 = FAISSVectorDB("adv2", str(tmp_path))
    db2.insert(["b"])

    (tmp_path / "invalid_dir").mkdir()

    result = FAISSVectorDB.list_adventures(str(tmp_path))

    assert set(result) == {"adv1", "adv2"}


def test_list_adventures_empty(tmp_path):
    """Empty directory should return empty list."""
    assert FAISSVectorDB.list_adventures(str(tmp_path)) == []


# =========================
# METADATA EDGE CASES
# =========================

def test_insert_metadata_padding(db):
    """Missing metadata should be auto-filled."""
    ids = db.insert(["a", "b"], metadata=[{"x": 1}])

    docs = db.get_by_ids(ids)

    assert docs[0]["metadata"]["x"] == 1
    assert "inserted_at" in docs[1]["metadata"]


def test_insert_metadata_truncation(db):
    """Extra metadata entries should be ignored."""
    ids = db.insert(["a"], metadata=[{"x": 1}, {"y": 2}])

    assert len(ids) == 1


# =========================
# CHUNKING LOGIC
# =========================

def test_chunk_query(db):
    """Chunking should split long queries correctly."""
    long_query = "word " * 100

    chunks = db._chunk_query(long_query, chunk_size=20)

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert "word" in chunks[0]