from src.llm_client import get_llm
from dotenv import load_dotenv


load_dotenv()

REWRITE_PROMPT = """You are helping improve search queries for a RAG system over Arabic bank \
procedure manuals (alarm systems, mail/files handling, warehouse operations).

The user's query below is vague, informal, or poorly phrased for search purposes. \
Rewrite it as a single clear, formal, search-friendly question in Arabic, preserving \
the original intent exactly — do not add information that wasn't implied by the original query.

Original query: {query}

Rewritten query:"""


def rewrite_query(query: str) -> str:
    llm = get_llm()
    response = llm.invoke(REWRITE_PROMPT.format(query=query))
    return response.content.strip()