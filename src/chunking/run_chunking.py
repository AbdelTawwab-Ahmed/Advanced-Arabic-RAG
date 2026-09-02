import os
import json
from src.chunking.chunker import chunk_document

PROCESSED_DIR = "data/processed"
CHUNKS_DIR = "data/processed/chunks"

if __name__ == "__main__":
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    md_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".md")]

    for md_file in md_files:
        path = os.path.join(PROCESSED_DIR, md_file)
        chunks = chunk_document(path)
        name = os.path.splitext(md_file)[0]

        # Human-readable version, for our own inspection
        debug_path = os.path.join(CHUNKS_DIR, f"{name}_chunks.md")
        with open(debug_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(f"=== page {c.page_number} | chunk {c.chunk_index} ===\n")
                f.write(c.text)
                f.write("\n\n")

        # Structured version, for the embedding stage to consume
        jsonl_path = os.path.join(CHUNKS_DIR, f"{name}_chunks.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        oversized_pages = len({c.page_number for c in chunks if c.chunk_index > 0})
        print(f"\n{md_file}: {len(chunks)} chunks -> {jsonl_path}")
        print(f"\n  → {oversized_pages} page(s) needed row-splitting")