import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import re


class FAISSVectorDB:
    """FAISS vector database for role-playing RAG system, per-adventure instance"""

    def __init__(self, adventure_name: str = "vanilla_fantasy", storage_path: str = "../adventure_memories"):
        """
        Initialize a vector database for a specific adventure.

        Args:
            adventure_name: Name of the adventure (used for directory name)
            storage_path: Base directory where vector databases are stored
        """
        self.adventure_name = adventure_name
        self.base_storage_path = Path(storage_path).resolve()  # Add .resolve() here

        # Create adventure-specific directory
        safe_name = self._sanitize_directory_name(adventure_name)
        self.adventure_path = self.base_storage_path / safe_name
        self.adventure_path.mkdir(parents=True, exist_ok=True)

        # Debug: Show where the vector DB is stored
        print(f"FAISS Vector DB for '{adventure_name}' stored at: {self.adventure_path}")

        # Set file paths
        self.index_path = self.adventure_path / "faiss_index"
        self.docs_path = self.adventure_path / "documents.pkl"
        self.metadata_path = self.adventure_path / "metadata.json"

        # Track if we need to rebuild index after deletions
        self._needs_rebuild = False

        # Initialize the embedding model (shared across all instances)
        self.embedding_model = self._load_embedding_model()

        # Initialize FAISS index and documents
        self.index = None
        self.documents = []
        self.metadata = {}
        self._load_or_create_index()

    def _sanitize_directory_name(self, name: str) -> str:
        """Convert adventure name to safe directory name."""
        # Replace unsafe characters with underscores
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        # Remove multiple underscores and trim
        safe = "_".join(filter(None, safe.split("_")))
        return safe

    def _load_embedding_model(self):
        """Load or download the embedding model."""
        model_path = self.base_storage_path / "models" / "all-MiniLM-L6-v2"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(model_path):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.save(str(model_path))
        else:
            model = SentenceTransformer(str(model_path))

        return model

    def _load_or_create_index(self):
        """Load existing index or create a new one for this adventure."""
        # Check for existing index file (without .faiss extension)
        if self.index_path.exists() and self.docs_path.exists():
            try:
                # Load existing FAISS index
                self.index = faiss.read_index(str(self.index_path))

                # Load documents
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

                # Load additional metadata if exists
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {}

                print(f"Loaded existing index for '{self.adventure_name}' with {len(self.documents)} documents")

                # Verify index and documents are in sync
                if self.index.ntotal != len(self.documents):
                    print(
                        f"Warning: Index size ({self.index.ntotal}) doesn't match documents count ({len(self.documents)})")

            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create a new FAISS index with 384 dimensions (all-MiniLM-L6-v2 output size)
        self.index = faiss.IndexFlatIP(384)  # Inner product (cosine similarity)
        self.documents = []
        self.metadata = {
            'adventure_name': self.adventure_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'total_documents': 0
        }
        print(f"Created new FAISS index for adventure: '{self.adventure_name}'")

        # Save the empty index immediately
        self.save()

    def save(self):
        """Save the index, documents, and metadata to disk."""
        # Update metadata
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['total_documents'] = len(self.documents)

        # Rebuild index if needed after deletions
        if self._needs_rebuild:
            self.rebuild_index()
            self._needs_rebuild = False

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save documents
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        print(f"Saved index for '{self.adventure_name}' with {len(self.documents)} documents")

    def insert(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[int]:
        """
        Insert texts into the vector database.

        Args:
            texts: List of text strings to insert
            metadata: Optional list of metadata dictionaries for each text

        Returns:
            List of document IDs that were inserted
        """
        if len(texts) == 0:
            return []

        # Ensure metadata list matches texts length
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) != len(texts):
            # Pad or truncate metadata to match texts
            if len(metadata) < len(texts):
                metadata = metadata + [{} for _ in range(len(texts) - len(metadata))]
            else:
                metadata = metadata[:len(texts)]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        embeddings = embeddings.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Get starting ID
        start_id = len(self.documents)

        # Add to index
        self.index.add(embeddings)

        # Store documents with metadata
        inserted_ids = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            doc_id = start_id + i
            self.documents.append({
                'id': doc_id,
                'text': text,
                'metadata': {
                    **meta,
                    'inserted_at': datetime.now().isoformat()
                },
                'created_at': datetime.now().isoformat(),
                'embedding_index': i  # Index within this batch
            })
            inserted_ids.append(doc_id)

        # Save after insertion
        self.save()

        print(f"Inserted {len(texts)} documents into '{self.adventure_name}'. Total: {len(self.documents)}")
        return inserted_ids

    def insert_single(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Insert a single text into the vector database.

        Args:
            text: Text string to insert
            metadata: Optional metadata dictionary

        Returns:
            Document ID that was inserted
        """
        return self.insert([text], [metadata] if metadata else [{}])[0]

    def delete(self, doc_ids: List[int]) -> bool:
        """
        Remove documents by ID.

        Args:
            doc_ids: List of document IDs to remove

        Returns:
            True if deletion was successful, False otherwise
        """
        if not doc_ids:
            return True

        try:
            # Filter out deleted documents
            initial_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc['id'] not in doc_ids]

            # Mark that we need to rebuild the index
            self._needs_rebuild = True

            print(f"Marked {initial_count - len(self.documents)} documents for deletion in '{self.adventure_name}'")
            return True

        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def _chunk_query(self, query: str, chunk_size: int = 200) -> List[str]:
        """
        Split a long query into smaller chunks for better search coverage.

        Args:
            query: The original query text
            chunk_size: Maximum number of words per chunk

        Returns:
            List of query chunks
        """
        if not query or chunk_size <= 0:
            return [query] if query else []

        # Split query into words
        words = query.split()

        if len(words) <= chunk_size:
            return [query]

        # Create chunks
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        # Ensure we don't lose context by adding overlapping chunks
        # This helps capture information that might be split at chunk boundaries
        if len(chunks) > 1:
            overlapping_chunks = []
            for i in range(len(chunks) - 1):
                # Create overlapping chunks with 25% overlap
                overlap_size = max(1, chunk_size // 4)
                if i == 0:
                    overlapping_chunks.append(chunks[i])

                # Create overlapping chunk between current and next
                current_words = chunks[i].split()
                next_words = chunks[i + 1].split()

                # Take last overlap_size words from current and first overlap_size from next
                overlap = ' '.join(current_words[-overlap_size:] + next_words[:overlap_size])
                overlapping_chunks.append(overlap)

                if i == len(chunks) - 2:
                    overlapping_chunks.append(chunks[i + 1])

            chunks = overlapping_chunks

        return chunks

    def search(self, query: str, k_per_cascade: int = 5, number_of_cascades: int = 1,
               threshold: float = 0.3, filter_metadata: Optional[Dict] = None,
               chunk_size: Optional[int] = None, debug: bool = False) -> List[Dict]:
        """
        Search for similar documents with optional cascade expansion.

        Args:
            query: Query text to search for
            k_per_cascade: Number of results to retrieve per cascade step
            number_of_cascades: Number of iterative search steps. If 1, performs a single search.
            threshold: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional metadata to filter results by
            chunk_size: If provided, split long queries into chunks of this size (in words).
                       If None or 0, use the entire query as is.

        Returns:
            List of unique result dictionaries from all cascade steps,
            sorted by similarity (highest first)
        """
        if self.index.ntotal == 0 or len(self.documents) == 0:
            print(f"Warning: Index for '{self.adventure_name}' is empty!")
            return []

        if debug:
            print(f"\n[DEBUG] Starting cascade search with {number_of_cascades} steps, k_per_cascade={k_per_cascade}, threshold={threshold}")

        # Single search (no cascade)
        if number_of_cascades <= 1:
            if chunk_size is not None and chunk_size > 0:
                return self._search_with_chunks(query, k_per_cascade, threshold, filter_metadata, chunk_size)
            else:
                return self._search_single_query(query, k_per_cascade, threshold, filter_metadata)

        # Cascade search
        all_results = []          # list of unique result dicts (in order of discovery)
        seen_ids = set()           # set of document IDs already added
        current_query = query

        for step in range(number_of_cascades):
            # Perform search for this step
            if chunk_size is not None and chunk_size > 0:
                step_results = self._search_with_chunks(
                    current_query, k_per_cascade, threshold, filter_metadata,
                    chunk_size, exclude_ids=seen_ids, debug=debug
                )
            else:
                step_results = self._search_single_query(
                    current_query, k_per_cascade, threshold, filter_metadata,
                    exclude_ids=seen_ids, debug=debug
                )

            if debug:
                print(f"\n[DEBUG] Cascade step {step + 1}: query = '{current_query[:100]}...'")
                print(f"[DEBUG] Retrieved {len(step_results)} documents (IDs: {[r['id'] for r in step_results]})")

            # Add new unique results
            new_ids = set()
            for res in step_results:
                if res['id'] not in seen_ids:
                    seen_ids.add(res['id'])
                    all_results.append(res)
                    new_ids.add(res['id'])

            # Prepare next query from the texts of the results just retrieved
            if step < number_of_cascades - 1:
                if not step_results:
                    # No results to continue, break early
                    break
                # Concatenate texts of the current step's results
                current_query = ' '.join([r['text'] for r in step_results])

        # Sort all unique results by similarity descending
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results

    def _search_single_query(self, query: str, k: int, threshold: float,
                             filter_metadata: Optional[Dict] = None,
                             exclude_ids: Optional[Set[int]] = None,
                             debug: bool = False) -> List[Dict]:
        """
        Search using a single query (original search logic).

        Args:
            query: Original query text
            k: Number of results to return
            threshold: Minimum similarity score
            filter_metadata: Optional metadata filter
            debug: If True, print debug information

        Returns:
            List of unique results sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search (request more than k to see beyond)
        distances, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))

        # Collect results
        results = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and distance >= threshold:
                if idx < len(self.documents):
                    doc = self.documents[idx]

                    # Apply metadata filter if specified
                    if filter_metadata:
                        if not self._matches_filter(doc['metadata'], filter_metadata):
                            continue

                    # Exclude already seen IDs
                    if exclude_ids and doc['id'] in exclude_ids:
                        continue

                    results.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'similarity': float(distance),
                        'id': doc['id'],
                        'created_at': doc.get('created_at')
                    })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        if debug:
            print(
                f"[DEBUG] _search_single_query: raw distances (first {min(5, len(distances[0]))}):\n{distances[0][:5]}\n{indices[0][:5]}")
            # Find the first below threshold or the (k+1)-th
            all_pairs = list(zip(distances[0], indices[0]))
            valid_pairs = [(d, idx) for d, idx in all_pairs if idx != -1]
            if results:
                top_id = results[0]['id']
                top_dist = results[0]['similarity']
                print(f"[DEBUG] Best match: ID {top_id} with similarity {top_dist:.4f}")
            # First candidate that was not included (either below threshold or excluded)
            not_chosen = None
            for d, idx in valid_pairs:
                if idx != -1 and d < threshold:
                    not_chosen = (idx, d)
                    break
            if not_chosen is None and len(valid_pairs) > len(results):
                not_chosen = valid_pairs[len(results)] if len(valid_pairs) > len(results) else None
            if not_chosen:
                doc = self.get_by_id(not_chosen[0])
                if doc:
                    print(
                        f"[DEBUG] Nearest not chosen: ID {not_chosen[0]} with similarity {not_chosen[1]:.4f} (text: {doc['text'][:80]}...)")
                else:
                    print(
                        f"[DEBUG] Nearest not chosen: ID {not_chosen[0]} with similarity {not_chosen[1]:.4f} (document not found?)")

        return results[:k]

    def _search_with_chunks(self, query: str, k: int, threshold: float,
                            filter_metadata: Optional[Dict] = None, chunk_size: int = 200,
                            exclude_ids: Optional[Set[int]] = None,
                            debug: bool = False) -> List[Dict]:
        """
        Search by splitting query into chunks and combining results,
        optionally excluding certain IDs.

        Args:
            query: Original query text
            k: Number of results to return from this chunked search
            threshold: Minimum similarity score
            filter_metadata: Optional metadata filter
            chunk_size: Words per chunk
            exclude_ids: Set of document IDs to exclude from results
            debug: If True, print debug information

        Returns:
            List of unique results sorted by similarity
        """
        if debug:
            print(f"[DEBUG] Searching with query chunking (chunk_size={chunk_size})")

        # Split query into chunks
        query_chunks = self._chunk_query(query, chunk_size)
        if debug:
            print(f"[DEBUG] Split query into {len(query_chunks)} chunks")

        # Use a set to track unique document IDs within this chunked search
        seen_ids_in_step = set()
        all_results = []

        # Search for each chunk
        for i, chunk in enumerate(query_chunks):
            if debug:
                print(f"[DEBUG] Searching chunk {i + 1}/{len(query_chunks)}: '{chunk[:50]}...'")

            # Get results for this chunk (request more than k to get diverse results)
            chunk_results = self._search_single_query(
                chunk, k * 3, threshold, filter_metadata,
                exclude_ids=exclude_ids, debug=debug
            )

            # Add unique results (avoid duplicates within this step)
            for result in chunk_results:
                if result['id'] not in seen_ids_in_step:
                    seen_ids_in_step.add(result['id'])
                    all_results.append(result)

        # Sort all results by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Return top k results
        final_results = all_results[:k]
        print(f"Found {len(final_results)} unique results from {len(query_chunks)} query chunks")
        return final_results
    def _matches_filter(self, doc_metadata: Dict, filter_metadata: Dict) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in doc_metadata or doc_metadata[key] != value:
                return False
        return True

    def get_by_id(self, doc_id: int) -> Optional[Dict]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def get_by_ids(self, doc_ids: List[int]) -> List[Dict]:
        """Get multiple documents by their IDs."""
        return [doc for doc in self.documents if doc['id'] in doc_ids]

    def get_all_documents(self, include_deleted: bool = False) -> List[Dict]:
        """
        Get all documents.

        Args:
            include_deleted: Whether to include documents marked as deleted

        Returns:
            List of all documents
        """
        if include_deleted:
            return self.documents
        else:
            # Filter out documents with deleted flag
            return [doc for doc in self.documents if not doc.get('deleted', False)]

    def get_document_count(self, include_deleted: bool = False) -> int:
        """Get total number of documents."""
        if include_deleted:
            return len(self.documents)
        else:
            return len([doc for doc in self.documents if not doc.get('deleted', False)])

    def get_index_size(self) -> int:
        """Get FAISS index size (total vectors)."""
        return self.index.ntotal

    def clear(self) -> bool:
        """
        Clear all documents and reset the index for this adventure.

        Returns:
            True if successful
        """
        try:
            # Reset to empty index
            self.documents = []
            self.index = faiss.IndexFlatIP(384)

            # Update metadata
            self.metadata['cleared_at'] = datetime.now().isoformat()
            self.metadata['total_documents'] = 0

            print(f"Cleared all documents for adventure: '{self.adventure_name}'")
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

    def delete_adventure(self) -> bool:
        """
        Delete the entire adventure database (all files).

        Returns:
            True if successful, False otherwise
        """
        try:
            # Close any open resources
            self.index = None

            # Delete all files in adventure directory
            for file_path in self.adventure_path.glob("*"):
                try:
                    file_path.unlink()
                except:
                    pass

            # Try to remove the directory
            if self.adventure_path.exists():
                self.adventure_path.rmdir()

            print(f"Deleted adventure database: '{self.adventure_name}'")
            return True

        except Exception as e:
            print(f"Error deleting adventure database: {e}")
            return False

    def export_documents(self, export_path: Optional[str] = None) -> str:
        """
        Export all documents to a JSON file.

        Args:
            export_path: Optional custom export path

        Returns:
            Path to the exported file
        """
        if export_path is None:
            export_path = self.adventure_path / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            'adventure_name': self.adventure_name,
            'export_timestamp': datetime.now().isoformat(),
            'total_documents': len(self.documents),
            'metadata': self.metadata,
            'documents': self.documents
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Exported {len(self.documents)} documents from '{self.adventure_name}' to {export_path}")
        return str(export_path)

    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = {
            'adventure_name': self.adventure_name,
            'total_documents': len(self.documents),
            'index_vectors': self.index.ntotal,
            'active_documents': self.get_document_count(include_deleted=False),
            'document_types': {},
            'last_update': self.metadata.get('updated_at'),
            'created_at': self.metadata.get('created_at')
        }

        # Count document types from metadata
        for doc in self.documents:
            if not doc.get('deleted', False):
                doc_type = doc.get('metadata', {}).get('type', 'unknown')
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1

        return stats

    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from current documents (useful after deletions).

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all active documents
            active_docs = [doc for doc in self.documents if not doc.get('deleted', False)]

            if not active_docs:
                # No documents, create empty index
                self.index = faiss.IndexFlatIP(384)
                self.documents = []
                return True

            # Collect texts
            texts = [doc['text'] for doc in active_docs]

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)

            # Create new index
            self.index = faiss.IndexFlatIP(384)
            self.index.add(embeddings)

            # Update document indices
            for i, doc in enumerate(active_docs):
                doc['embedding_index'] = i

            print(f"Rebuilt index for '{self.adventure_name}' with {len(active_docs)} documents")
            return True

        except Exception as e:
            print(f"Error rebuilding index: {e}")
            return False

    def close(self):
        """Close the database (save before closing)."""
        self.save()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save and close."""
        self.close()

    @classmethod
    def list_adventures(cls, storage_path: str = "../adventure_memories") -> List[str]:
        """
        List all adventure databases in the storage directory.

        Args:
            storage_path: Base directory to search

        Returns:
            List of adventure names
        """
        path = Path(storage_path)
        if not path.exists():
            return []

        adventures = []
        for item in path.iterdir():
            if item.is_dir():
                # Check if it has FAISS files (without .faiss extension)
                index_path = item / "faiss_index"
                docs_path = item / "documents.pkl"
                if index_path.exists() and docs_path.exists():
                    adventures.append(item.name)

        return adventures

if __name__ == "__main__":
    import argparse
    import shlex
    import sys
    from pathlib import Path

    # Configuration
    DEFAULT_ADVENTURE = "vanilla_fantasy"
    STORAGE_PATH = "../adventure_memories"

    # Parse adventure name from command line (optional)
    adventure_name = DEFAULT_ADVENTURE
    if len(sys.argv) > 1:
        adventure_name = sys.argv[1]

    # Build expected paths
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in adventure_name)
    safe_name = "_".join(filter(None, safe_name.split("_")))
    adventure_dir = Path(STORAGE_PATH).resolve() / safe_name
    index_path = adventure_dir / "faiss_index"
    docs_path = adventure_dir / "documents.pkl"

    # Check if database exists
    if not (index_path.exists() and docs_path.exists()):
        print(f"Error: Adventure database '{adventure_name}' does not exist at {adventure_dir}")
        print("Available adventures:")
        available = FAISSVectorDB.list_adventures(STORAGE_PATH)
        for adv in available:
            print(f"  {adv}")
        sys.exit(1)

    # Create database instance (will load existing)
    db = FAISSVectorDB(adventure_name=adventure_name, storage_path=STORAGE_PATH)

    def print_help():
        print("\nAvailable commands:")
        print("  help                                  Show this help")
        print("  list [--limit N] [--offset N]         List documents (with pagination)")
        print("  search <query> [--k N] [--threshold F] [--cascades N] [--chunk-size N] [--filter key=val ...]")
        print("  stats                                  Show database statistics")
        print("  insert <text> [--metadata key=val ...] Insert a single document")
        print("  delete <id>                            Delete document by ID")
        print("  exit                                   Exit the program")
        print()

    def parse_key_value_pairs(args):
        """Convert list of 'key=value' strings into dict."""
        result = {}
        for item in args:
            if '=' in item:
                k, v = item.split('=', 1)
                result[k] = v
            else:
                print(f"Warning: ignoring malformed metadata '{item}' (expected key=value)")
        return result

    print(f"FAISS Vector DB CLI for adventure '{adventure_name}'")
    print_help()

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            # Use shlex to split respecting quotes
            parts = shlex.split(line)
            command = parts[0].lower()

            if command == "exit":
                break

            elif command == "help":
                print_help()

            elif command == "list":
                # Parse optional --limit and --offset
                limit = None
                offset = None
                i = 1
                while i < len(parts):
                    if parts[i] == "--limit" and i + 1 < len(parts):
                        limit = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "--offset" and i + 1 < len(parts):
                        offset = int(parts[i + 1])
                        i += 2
                    else:
                        print(f"Unknown option: {parts[i]}")
                        break
                all_docs = db.get_all_documents()
                if offset is not None:
                    all_docs = all_docs[offset:]
                if limit is not None:
                    all_docs = all_docs[:limit]
                for doc in all_docs:
                    print(f"ID: {doc['id']} | {doc['text'][:80]}..." if len(doc['text']) > 80 else doc['text'])
                print(f"Total shown: {len(all_docs)} / {db.get_document_count()}")

            elif command == "search":
                if len(parts) < 2:
                    print("Error: missing query. Usage: search <query> [options]")
                    continue
                query = parts[1]
                k = 5
                threshold = 0.3
                cascades = 1
                chunk_size = None
                filter_metadata = {}
                debug = False
                i = 2
                while i < len(parts):
                    if parts[i] == "--k" and i + 1 < len(parts):
                        k = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "--threshold" and i + 1 < len(parts):
                        threshold = float(parts[i + 1])
                        i += 2
                    elif parts[i] == "--cascades" and i + 1 < len(parts):
                        cascades = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "--chunk-size" and i + 1 < len(parts):
                        chunk_size = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "--filter":
                        i += 1
                        filters = []
                        while i < len(parts) and '=' in parts[i] and not parts[i].startswith('--'):
                            filters.append(parts[i])
                            i += 1
                        filter_metadata = parse_key_value_pairs(filters)
                    elif parts[i] == "--debug":
                        debug = True
                        i += 1
                    else:
                        print(f"Unknown option: {parts[i]}")
                        break
                results = db.search(query, k_per_cascade=k, number_of_cascades=cascades,
                                    threshold=threshold, filter_metadata=filter_metadata,
                                    chunk_size=chunk_size, debug=debug)
                print(f"\nFound {len(results)} results:")
                for r in results:
                    print(f"ID: {r['id']} | Similarity: {r['similarity']:.3f} | {r['text'][:80]}...")
                print()

            elif command == "stats":
                stats = db.get_stats()
                for key, value in stats.items():
                    print(f"{key}: {value}")

            elif command == "insert":
                if len(parts) < 2:
                    print("Error: missing text. Usage: insert <text> [--metadata key=val ...]")
                    continue
                text = parts[1]
                metadata = {}
                i = 2
                while i < len(parts):
                    if parts[i] == "--metadata":
                        i += 1
                        meta_items = []
                        while i < len(parts) and '=' in parts[i] and not parts[i].startswith('--'):
                            meta_items.append(parts[i])
                            i += 1
                        metadata = parse_key_value_pairs(meta_items)
                    else:
                        print(f"Unknown option: {parts[i]}")
                        break
                doc_id = db.insert_single(text, metadata)
                print(f"Inserted document with ID: {doc_id}")

            elif command == "delete":
                if len(parts) < 2:
                    print("Error: missing document ID. Usage: delete <id>")
                    continue
                try:
                    doc_id = int(parts[1])
                except ValueError:
                    print("Error: document ID must be an integer.")
                    continue
                success = db.delete([doc_id])
                if success:
                    print(f"Deleted document {doc_id}")
                    db.save()
                else:
                    print(f"Failed to delete document {doc_id}")

            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")