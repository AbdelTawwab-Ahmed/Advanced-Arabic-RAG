import os
import time
from typing import List
import cohere
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv


load_dotenv()

# --- Dense (Embedding | Semantic Vectors) ---

CO = cohere.Client(os.getenv("COHERE_API_KEY"))
EMBED_MODEL = "embed-multilingual-v3.0"
BATCH_SIZE = 96  


def embed_dense(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """
    Generates dense (semantic) embeddings for a list of texts using Cohere.
    input_type must be 'search_document' when embedding chunks for indexing,
    and 'search_query' when embedding a user's query at retrieval time.
    """
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = CO.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type=input_type,
        )
        all_embeddings.extend(response.embeddings)
        time.sleep(0.5)  
    return all_embeddings


# --- Sparse (Keywords Search | BM25) ---


SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_sparse(texts: List[str]):
    """
    Generates sparse (BM25-style) embeddings for a list of texts using FastEmbed.
    Runs locally, no API call — returns FastEmbed's SparseEmbedding objects,
    each with .indices and .values, which map directly to Qdrant's sparse vector format.
    """
    return list(SPARSE_MODEL.embed(texts))