import json
import uuid
from typing import List, Iterator
from qdrant_client.models import PointStruct, SparseVector

from src.embedding.embedder import embed_dense, embed_sparse
from src.indexing.qdrant_setup import get_qdrant_client, create_collection, COLLECTION_NAME

UPSERT_BATCH_SIZE = 50


def deterministic_uuid(chunk_id: str) -> str:
    """Same chunk_id always produces the same UUID, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def load_chunks(jsonl_path: str) -> List[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def batched(items: List[dict], size: int) -> Iterator[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def index_jsonl_file(jsonl_path: str):
    client = get_qdrant_client()
    create_collection(client)  

    chunks = load_chunks(jsonl_path)
    print(f"\nLoaded {len(chunks)} chunks from {jsonl_path}")

    for batch in batched(chunks, UPSERT_BATCH_SIZE):
        texts = [c["text"] for c in batch]

        dense_vectors = embed_dense(texts, input_type="search_document")
        sparse_vectors = embed_sparse(texts)

        points = []
        for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
            points.append(PointStruct(
                id=deterministic_uuid(chunk["id"]),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "source_pdf": chunk["source_pdf"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                },
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\n  Upserted batch of {len(points)} points")