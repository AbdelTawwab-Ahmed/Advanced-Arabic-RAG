from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

HYDE_PROMPT = """You are writing a plausible-sounding excerpt from a bank's internal Arabic \
procedure manual (covering alarm systems, mail/files handling, warehouse operations), \
answering the question below.

Write ONE short paragraph, in formal Arabic, in the same declarative style these manuals use \
(stating procedures/facts directly, not as a Q&A). Do not include disclaimers, hedging, or \
mention that this is hypothetical — write it as if it were a real excerpt.

Question: {query}
"""


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def generate_hyde_document(query: str) -> str:
    llm = get_llm()
    response = llm.invoke(HYDE_PROMPT.format(query=query))
    return response.content.strip()