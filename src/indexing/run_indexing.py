import os
from src.indexing.indexer import index_jsonl_file

CHUNKS_DIR = "data/processed/chunks"

if __name__ == "__main__":
    jsonl_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".jsonl")]
    for jsonl_file in jsonl_files:
        path = os.path.join(CHUNKS_DIR, jsonl_file)
        print(f"\n=== Start Indexing {jsonl_file} ===")
        index_jsonl_file(path)
        print(f"\n=== Finish Indexing {jsonl_file} ===")