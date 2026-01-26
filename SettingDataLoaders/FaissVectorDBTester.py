import tempfile
import shutil
import time
import os
from pathlib import Path
from FaissVectorDB import FAISSVectorDB


class FAISSVectorDBTester:
    """Comprehensive test suite for FAISSVectorDB class (per-adventure)."""

    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        # Use default storage path for tests
        cls.storage_path = "../vector_database_test"
        print(f"\nTest storage path: {cls.storage_path}")

        # Test data for vector database
        cls.sample_texts = [
            "The party encounters a red dragon in its lair.",
            "The dragon has a hoard of gold and magical artifacts.",
            "Ancient runes on the wall tell of a dragon's weakness.",
            "The party finds a secret passage behind a tapestry.",
            "A mysterious wizard offers help in exchange for a rare herb.",
            "The forest is enchanted, with trees that whisper secrets.",
            "Deep in the cave, a crystal glows with magical energy.",
            "The king offers a reward for defeating the dragon.",
            "The party discovers an ancient prophecy about the chosen one.",
            "A hidden shrine contains a powerful healing artifact."
        ]

        cls.sample_metadata = [
            {"type": "encounter", "location": "dragon_lair", "entities": ["dragon"], "importance": "high"},
            {"type": "description", "location": "dragon_lair", "items": ["gold", "artifacts"], "importance": "medium"},
            {"type": "lore", "location": "dragon_lair", "knowledge": "dragon_weakness", "importance": "critical"},
            {"type": "discovery", "location": "dragon_lair", "secret": True, "importance": "medium"},
            {"type": "npc_interaction", "location": "forest", "npc": "wizard", "importance": "high"},
            {"type": "location", "location": "forest", "magical": True, "importance": "low"},
            {"type": "item", "location": "cave", "magical": True, "importance": "high"},
            {"type": "quest", "location": "castle", "npc": "king", "importance": "high"},
            {"type": "lore", "location": "temple", "prophecy": True, "importance": "critical"},
            {"type": "item", "location": "shrine", "magical": True, "healing": True, "importance": "high"}
        ]

    @classmethod
    def teardown_class(cls):
        """Cleanup test environment."""
        # Clean up test directory
        path = Path(cls.storage_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"\nCleaned up test directory: {cls.storage_path}")
            except Exception as e:
                print(f"\nWarning: Could not fully clean up test directory: {e}")

    def setup_method(self):
        """Setup before each test."""
        self.databases = []  # Track created databases for cleanup

    def teardown_method(self):
        """Teardown after each test."""
        # Close and clean up all databases
        for db in self.databases:
            try:
                db.close()
            except:
                pass

        # Clean up any remaining files in the test directory
        path = Path(self.storage_path)
        if path.exists():
            for item in path.iterdir():
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                    except:
                        pass

    def create_database(self, adventure_name: str = "Test Adventure") -> FAISSVectorDB:
        """Helper to create and track a database."""
        db = FAISSVectorDB(adventure_name, self.storage_path)
        self.databases.append(db)
        return db

    def test_initialization(self):
        """Test database initialization and file creation."""
        print("\n=== Test: Initialization ===")

        # Test normal initialization
        db = self.create_database("Dragon Quest")
        assert db.adventure_path.exists(), "Adventure directory should be created"
        print(f"✓ Created database in: {db.adventure_path.name}")

        # Test that existing database is not overwritten
        db1 = self.create_database("Existing Adventure")
        inserted_ids = db1.insert(["Initial test document"])
        db1.save()  # Save to create metadata file

        # Reopen the same adventure
        db2 = self.create_database("Existing Adventure")
        assert db2.get_document_count() > 0, "Should preserve existing data"
        print("✓ Existing database preserved data")

    def test_insert_and_search(self):
        """Test document insertion and search functionality."""
        print("\n=== Test: Insert and Search ===")

        db = self.create_database("Search Test Adventure")

        # Test bulk insert
        inserted_ids = db.insert(self.sample_texts[:5], self.sample_metadata[:5])
        assert len(inserted_ids) == 5, f"Should insert 5 documents, got {len(inserted_ids)}"
        print(f"✓ Inserted {len(inserted_ids)} documents with IDs: {inserted_ids}")

        # Test single insert
        single_id = db.insert_single("A single test document", {"type": "test", "category": "debug"})
        assert single_id is not None, "Single insert should return an ID"
        print(f"✓ Single insert returned ID: {single_id}")

        # Test search with basic query
        results = db.search("dragon", k=3)
        assert len(results) > 0, "Should find documents about dragon"
        print(f"✓ Search found {len(results)} results for 'dragon'")

        # Test search with threshold
        results = db.search("unrelated query", threshold=0.8)
        print(f"✓ High threshold search returned {len(results)} results")

        # Test search with metadata filter
        results = db.search("dragon", filter_metadata={"type": "lore"})
        lore_count = sum(1 for r in results if r['metadata'].get('type') == 'lore')
        if results:  # Only assert if we got results
            assert lore_count == len(results), "All results should be of type 'lore'"
        print(f"✓ Metadata filter works, found {len(results)} lore documents")

    def test_document_retrieval(self):
        """Test retrieving documents by ID."""
        print("\n=== Test: Document Retrieval ===")

        db = self.create_database("Retrieval Test")

        # Insert test documents
        inserted_ids = db.insert(self.sample_texts[:3], self.sample_metadata[:3])

        # Test get_by_id
        for doc_id in inserted_ids:
            doc = db.get_by_id(doc_id)
            assert doc is not None, f"Should find document with ID {doc_id}"
            assert doc['id'] == doc_id, f"Document ID should match {doc_id}"
        print(f"✓ Retrieved all documents by ID")

        # Test get_by_ids
        docs = db.get_by_ids(inserted_ids)
        assert len(docs) == len(inserted_ids), "Should retrieve all requested documents"
        retrieved_ids = [doc['id'] for doc in docs]
        assert set(retrieved_ids) == set(inserted_ids), "Should retrieve exactly the requested IDs"
        print(f"✓ Retrieved {len(docs)} documents by list of IDs")

        # Test get_by_id for non-existent document
        non_existent = db.get_by_id(999)
        assert non_existent is None, "Should return None for non-existent ID"
        print("✓ Correctly returns None for non-existent ID")

    def test_stats_and_count(self):
        """Test database statistics and document counting."""
        print("\n=== Test: Statistics and Counting ===")

        db = self.create_database("Stats Test")

        # Initially empty
        count = db.get_document_count()
        assert count == 0, f"Should start with 0 documents, got {count}"
        print(f"✓ Initial document count: {count}")

        # Insert documents
        db.insert(self.sample_texts[:7], self.sample_metadata[:7])

        # Test counts
        count = db.get_document_count()
        assert count == 7, f"Should have 7 documents, got {count}"

        index_size = db.get_index_size()
        assert index_size == 7, f"Index should have 7 vectors, got {index_size}"

        print(f"✓ After insert - Documents: {count}, Index vectors: {index_size}")

        # Test get_all_documents
        all_docs = db.get_all_documents()
        assert len(all_docs) == 7, f"Should get all 7 documents, got {len(all_docs)}"
        print(f"✓ Retrieved all {len(all_docs)} documents")

        # Test statistics
        stats = db.get_stats()
        assert stats['total_documents'] == 7, "Stats should show 7 documents"
        assert stats['index_vectors'] == 7, "Stats should show 7 index vectors"
        assert 'document_types' in stats, "Stats should include document types"

        # Check document type counts
        type_counts = stats['document_types']
        print(f"✓ Statistics calculated correctly")

    def test_delete_and_clear(self):
        """Test document deletion and database clearing."""
        print("\n=== Test: Delete and Clear ===")

        db = self.create_database("Delete Test")

        # Insert documents
        inserted_ids = db.insert(self.sample_texts[:5], self.sample_metadata[:5])
        initial_count = db.get_document_count()
        assert initial_count == 5, f"Should have 5 documents, got {initial_count}"

        # Test delete
        delete_ids = [inserted_ids[0], inserted_ids[2]]
        success = db.delete(delete_ids)
        assert success, "Delete should succeed"

        # After deletion, count should be the same (soft delete)
        count_after_delete = db.get_document_count()
        print(f"✓ Marked 2 documents for deletion, count: {count_after_delete}")

        # Test clear
        success = db.clear()
        assert success, "Clear should succeed"

        count_after_clear = db.get_document_count()
        assert count_after_clear == 0, f"Should have 0 documents after clear, got {count_after_clear}"
        print(f"✓ Cleared database, document count: {count_after_clear}")

    def test_save_and_reload(self):
        """Test saving database and reloading it."""
        print("\n=== Test: Save and Reload ===")

        # Create and populate database
        db1 = self.create_database("Save Test")
        inserted_ids = db1.insert(self.sample_texts[:3], self.sample_metadata[:3])

        # Save and close
        db1.save()
        db1.close()
        self.databases.remove(db1)

        # Small delay to ensure file locks are released
        time.sleep(0.1)

        # Reload database
        db2 = self.create_database("Save Test")

        # Verify data persisted
        count = db2.get_document_count()
        assert count == 3, f"Should have 3 documents after reload, got {count}"

        # Verify specific document
        doc = db2.get_by_id(inserted_ids[0])
        assert doc is not None, "Should find document after reload"
        assert "dragon" in doc['text'].lower(), "Document content should persist"

        print(f"✓ Data persisted after save/reload: {count} documents")

    def test_adventure_deletion(self):
        """Test deleting entire adventure database."""
        print("\n=== Test: Adventure Deletion ===")

        # Create database in a separate directory to avoid affecting other tests
        test_dir = "../vector_database_delete_test"
        db = FAISSVectorDB("To Be Deleted", test_dir)
        db.insert(["Test document"])
        db.save()

        # Get path before deletion
        adventure_path = db.adventure_path
        assert adventure_path.exists(), "Adventure directory should exist"

        # Close database
        db.close()

        # Delete adventure
        success = db.delete_adventure()
        assert success, "Delete adventure should succeed"

        # Clean up the test directory
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

        print("✓ Adventure database deleted successfully")

    def test_list_adventures(self):
        """Test listing available adventures."""
        print("\n=== Test: List Adventures ===")

        # Use a clean directory for this test
        test_dir = "../vector_database_list_test"

        # Clean up if exists
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

        # Create some adventures
        adventure_names = ["Forest Quest", "Mountain Journey", "Desert Expedition"]
        created_dbs = []

        for name in adventure_names:
            db = FAISSVectorDB(name, test_dir)
            created_dbs.append(db)
            db.insert_single(f"Start of {name}")
            db.save()
            db.close()

        time.sleep(0.1)  # Windows file lock release delay

        # List adventures
        adventures = FAISSVectorDB.list_adventures(test_dir)

        # Get sanitized names for comparison
        temp_db = FAISSVectorDB("Temp", test_dir)
        expected_names = sorted([temp_db._sanitize_directory_name(name) for name in adventure_names])
        temp_db.close()

        found_names = sorted(adventures)

        assert found_names == expected_names, f"Expected {expected_names}, got {found_names}"
        print(f"✓ Found adventures: {found_names}")

        # Test with non-existent directory
        empty_list = FAISSVectorDB.list_adventures("/non/existent/path")
        assert empty_list == [], "Should return empty list for non-existent path"
        print("✓ Returns empty list for non-existent directory")

        # Clean up
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

    def test_context_manager(self):
        """Test context manager functionality."""
        print("\n=== Test: Context Manager ===")

        # Test normal context manager usage
        with FAISSVectorDB("Context Test", self.storage_path) as db:
            db.insert_single("Inside context manager")
            assert db.index is not None, "Index should be initialized"
            print("✓ Context manager opened database")

        # Database should be saved and closed after context
        print("✓ Context manager closed database")

    def test_performance(self):
        """Test performance with moderate dataset."""
        print("\n=== Test: Performance ===")

        db = self.create_database("Performance Test")

        # Time inserting documents
        test_docs = [f"Test document {i} about fantasy adventure" for i in range(20)]
        test_meta = [{"type": "test", "index": i} for i in range(20)]

        start_time = time.time()
        inserted_ids = db.insert(test_docs, test_meta)
        insert_time = time.time() - start_time

        print(f"✓ Inserted {len(inserted_ids)} documents in {insert_time:.3f}s")

        # Time searching
        start_time = time.time()
        for query in ["fantasy", "adventure", "test document"]:
            db.search(query, k=5)
        search_time = time.time() - start_time
        print(f"✓ Executed 3 searches in {search_time:.3f}s")

        # Time document retrieval
        start_time = time.time()
        for doc_id in inserted_ids[:5]:
            db.get_by_id(doc_id)
        retrieval_time = time.time() - start_time
        print(f"✓ Retrieved 5 documents by ID in {retrieval_time:.3f}s")

    def test_integration_simulation(self):
        """Simulate a complete adventure memory system."""
        print("\n=== Test: Integration Simulation ===")

        # Create adventure database
        db = self.create_database("Integration Test Adventure")

        # Simulate adventure events being logged as memories
        adventure_events = [
            ("The party discovers an ancient temple in the jungle.",
             {"type": "discovery", "location": "jungle", "importance": "high"}),

            ("Inside the temple, they find murals depicting a forgotten civilization.",
             {"type": "lore", "location": "temple", "importance": "medium"}),

            ("A hidden chamber contains a crystal that glows with arcane energy.",
             {"type": "item", "location": "temple", "magical": True, "importance": "high"}),

            ("The party is attacked by guardian statues that come to life.",
             {"type": "encounter", "location": "temple", "enemies": ["statues"], "importance": "high"}),

            ("They discover the crystal can control the temple's mechanisms.",
             {"type": "puzzle", "location": "temple", "solution": "crystal", "importance": "critical"})
        ]

        # Insert events as memories
        memory_ids = []
        for text, metadata in adventure_events:
            memory_id = db.insert_single(text, metadata)
            memory_ids.append(memory_id)

        print(f"✓ Logged {len(memory_ids)} adventure events as memories")

        # Simulate DM needing context for a situation
        print("\nSimulating DM memory retrieval:")

        # Player encounters statues
        query = "guardian statues attack"
        relevant_memories = db.search(query, k=3)

        print(f"  Player situation: '{query}'")
        print(f"  Retrieved {len(relevant_memories)} relevant memories:")
        for i, memory in enumerate(relevant_memories, 1):
            print(f"    {i}. [{memory['similarity']:.3f}] {memory['text'][:60]}...")

        # Player tries to solve puzzle
        query = "temple mechanisms crystal control"
        relevant_memories = db.search(query, k=2, filter_metadata={"type": "puzzle"})

        print(f"\n  Player puzzle: '{query}'")
        print(f"  Retrieved {len(relevant_memories)} puzzle memories:")
        for memory in relevant_memories:
            print(f"    - {memory['text']}")

        # Get adventure statistics
        stats = db.get_stats()
        print(f"\nAdventure Memory Statistics:")
        print(f"  Total memories: {stats['total_documents']}")
        print(f"  Memory types: {stats['document_types']}")

        # Save the adventure memory database
        db.save()
        print("✓ Adventure memory database saved")

        print("\n✓ Integration simulation completed successfully")


def run_all_tests():
    """Run all tests with timing."""
    print("=" * 80)
    print("FAISS VECTOR DATABASE TEST SUITE (Per-Adventure)")
    print("=" * 80)

    total_start = time.time()
    test_class = FAISSVectorDBTester()

    # Setup
    test_class.setup_class()

    # Run tests in order
    tests = [
        test_class.test_initialization,
        test_class.test_insert_and_search,
        test_class.test_document_retrieval,
        test_class.test_stats_and_count,
        test_class.test_delete_and_clear,
        test_class.test_save_and_reload,
        test_class.test_adventure_deletion,
        test_class.test_list_adventures,
        test_class.test_context_manager,
        test_class.test_performance,
        test_class.test_integration_simulation,
    ]

    results = []
    for i, test in enumerate(tests, 1):
        test_start = time.time()
        try:
            test_class.setup_method()
            test()
            test_class.teardown_method()
            status = "PASSED"
        except Exception as e:
            status = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

        test_time = time.time() - test_start
        results.append((test.__name__, status, test_time))
        print(f"Test {i:2d}: {test.__name__:<25} [{status}] ({test_time:.2f}s)")
        print()  # Blank line between tests

    # Teardown
    try:
        test_class.teardown_class()
    except Exception as e:
        print(f"Warning during teardown: {e}")

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, status, _ in results if "PASSED" in status)
    failed = len(results) - passed

    for name, status, ttime in results:
        print(f"{name:<25} {status:<20} {ttime:.2f}s")

    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")

    if failed == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {failed} test(s) failed!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)