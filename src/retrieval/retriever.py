from typing import List, Literal
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

from src.embedding.embedder import embed_dense, embed_sparse
from src.indexing.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from src.query_transformation.multi_query import dedupe_chunks
from src import config


InputType = Literal["search_query", "search_document"]


def _point_to_chunk(point) -> dict:
    return {
        "chunk_id": point.payload["chunk_id"],
        "score": point.score,
        "text": point.payload["text"],
        "source_pdf": point.payload["source_pdf"],
        "page_number": point.payload["page_number"],
    }


def retrieve_single(text: str, top_k: int = config.RETRIEVAL_TOP_K_SINGLE, input_type: InputType = "search_query") -> List[dict]:
    """
    Hybrid (dense + sparse, RRF-fused) retrieval for one piece of text.
    input_type matters: use 'search_query' for a real user question, but
    'search_document' when embedding a HyDE hypothetical document — see note below.
    """
    client = get_qdrant_client()

    dense_vec = embed_dense([text], input_type=input_type)[0]
    sparse_vec = embed_sparse([text])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    return [_point_to_chunk(p) for p in results.points]


def retrieve_multi(texts: List[str], top_k_per_query: int = config.RETRIEVAL_TOP_K_PER_SUBQUERY) -> List[dict]:
    """
    Retrieves for each text independently, then merges into one deduplicated,
    score-sorted list. Used for Multi-Query (several phrasings of one intent)
    and Decomposition (several distinct sub-questions).
    """
    all_results = [retrieve_single(t, top_k=top_k_per_query) for t in texts]
    return dedupe_chunks(all_results)