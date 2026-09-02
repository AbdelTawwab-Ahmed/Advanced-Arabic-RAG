import os
from typing import Literal, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

load_dotenv()

Technique = Literal["none", "rewriting", "multi_query", "decomposition", "hyde"]


class RouteDecision(BaseModel):
    technique: Technique = Field(description="The single best-fit technique for this query.")
    reason: str = Field(description="One short sentence explaining why this technique fits.")


ROUTER_PROMPT = """You are a query router for a RAG system over Arabic bank procedure manuals \
(alarm systems, mail/files handling, warehouse operations).

Classify the user's query into exactly ONE of these techniques:

- none: simple, self-contained, directly answerable by looking up one clear fact or step.
- rewriting: vague, informal, or poorly phrased in a way that would hurt retrieval if searched as-is.
- multi_query: could reasonably be phrased several different valid ways; retrieval would benefit from searching multiple phrasings.
- decomposition: asks about multiple distinct sub-topics or steps that should be retrieved separately, then combined.
- hyde: abstract or conceptual, hard to match against literal document wording; a hypothetical answer would retrieve better than the literal question.

Query: {query}
"""


class RouterState(TypedDict):
    query: str
    technique: Optional[str]
    reason: Optional[str]


def get_router_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def classify_node(state: RouterState) -> RouterState:
    llm = get_router_llm().with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
    return {**state, "technique": decision.technique, "reason": decision.reason}


def build_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classify", classify_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", END)
    return graph.compile()