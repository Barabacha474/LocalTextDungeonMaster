import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
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

    def search(self, query: str, k: int = 5, threshold: float = 0.3,
               filter_metadata: Optional[Dict] = None, chunk_size: Optional[int] = None) -> List[Dict]:
        """
        Search for similar documents with optional query chunking.

        Args:
            query: Query text to search for
            k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional metadata to filter results by
            chunk_size: If provided, split long queries into chunks of this size (in words).
                       If None or 0, use the entire query as is.

        Returns:
            List of result dictionaries sorted by similarity (highest first)
        """
        if self.index.ntotal == 0 or len(self.documents) == 0:
            print(f"Warning: Index for '{self.adventure_name}' is empty!")
            return []

        # If chunk_size is specified and query is long enough, split into chunks
        if chunk_size is not None and chunk_size > 0:
            return self._search_with_chunks(query, k, threshold, filter_metadata, chunk_size)

        # Original search logic for single query
        return self._search_single_query(query, k, threshold, filter_metadata)

    def _search_single_query(self, query: str, k: int, threshold: float,
                             filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search using a single query (original search logic)."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        # Collect results
        results = []
        print("MESSAGE FROM VECTOR DB SEARCH (single query)")
        print(f"threshold: {threshold}\n")

        for distance, idx in zip(distances[0], indices[0]):
            print(f"distance: {distance}\n")
            if idx != -1 and distance >= threshold:
                # Ensure idx is within current documents list bounds
                if idx < len(self.documents):
                    doc = self.documents[idx]

                    # Apply metadata filter if specified
                    if filter_metadata:
                        if not self._matches_filter(doc['metadata'], filter_metadata):
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
        return results[:k]

    def _search_with_chunks(self, query: str, k: int, threshold: float,
                            filter_metadata: Optional[Dict] = None, chunk_size: int = 200) -> List[Dict]:
        """
        Search by splitting query into chunks and combining results.

        Args:
            query: Original query text
            k: Number of final results to return
            threshold: Minimum similarity score
            filter_metadata: Optional metadata filter
            chunk_size: Words per chunk

        Returns:
            List of unique results sorted by similarity
        """
        print(f"Searching with query chunking (chunk_size={chunk_size})")

        # Split query into chunks
        query_chunks = self._chunk_query(query, chunk_size)
        print(f"Split query into {len(query_chunks)} chunks")

        # Use a set to track unique document IDs
        seen_ids = set()
        all_results = []

        # Search for each chunk
        for i, chunk in enumerate(query_chunks):
            print(f"Searching chunk {i + 1}/{len(query_chunks)}: '{chunk[:50]}...'")

            # Get results for this chunk (request more than k to get diverse results)
            chunk_results = self._search_single_query(chunk, k * 2, threshold, filter_metadata)

            # Add unique results
            for result in chunk_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
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
    vector_db = FAISSVectorDB()
    print(vector_db.get_stats())