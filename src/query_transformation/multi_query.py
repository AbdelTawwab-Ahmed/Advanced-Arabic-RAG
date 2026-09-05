from typing import List
from src.llm_client import get_llm
from src import config
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

MULTI_QUERY_PROMPT = """You are helping a RAG system over Arabic bank procedure manuals \
(alarm systems, mail/files handling, warehouse operations) retrieve better results.

Generate exactly 3 different Arabic phrasings of the query below, each capturing the same \
intent but varying in wording, terminology, or structure — so that searching with all 3 \
increases the chance of matching how the answer is actually phrased in the source documents. \
Do not add new intent or scope beyond the original query.

Original query: {query}
"""


class QueryVariants(BaseModel):
    variants: List[str] = Field(description="Exactly 3 differently-phrased Arabic versions of the query.")


def generate_multi_queries(query: str) -> List[str]:
    llm = get_llm(temperature=config.TEMP_VARIED).with_structured_output(QueryVariants)
    result: QueryVariants = llm.invoke(MULTI_QUERY_PROMPT.format(query=query))
    return result.variants


def dedupe_chunks(chunk_lists: List[List[dict]]) -> List[dict]:
    """Merges multiple retrieval result lists into one deduplicated list, keyed by chunk_id.
    Keeps the highest score seen for any chunk that appeared under multiple query variants."""
    merged = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid not in merged or chunk["score"] > merged[cid]["score"]:
                merged[cid] = chunk
    return sorted(merged.values(), key=lambda c: c["score"], reverse=True)