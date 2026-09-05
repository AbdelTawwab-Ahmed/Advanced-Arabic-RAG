import cohere
from typing import List
from src import config


CO = cohere.Client(config.COHERE_API_KEY)
RERANK_MODEL = config.COHERE_RERANK_MODEL


def rerank_chunks(query: str, chunks: List[dict], top_n: int = config.RERANK_TOP_N) -> List[dict]:
    """
    Reranks retrieved chunks against the ORIGINAL user query (not a transformed
    version — see note below), returning the top_n most relevant, with each
    chunk's dict updated to include a 'rerank_score'.
    """
    if not chunks:
        return []

    documents = [c["text"] for c in chunks]
    response = CO.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(chunks)),
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)
    return reranked