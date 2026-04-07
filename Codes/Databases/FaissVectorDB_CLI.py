from FaissVectorDB import FAISSVectorDB

if __name__ == "__main__":
    import argparse
    import shlex
    import sys
    from pathlib import Path

    # Configuration
    DEFAULT_ADVENTURE = "vanilla_fantasy"
    STORAGE_PATH = "../../adventure_memories"

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