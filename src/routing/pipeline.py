from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from src.routing.router import RouteDecision, ROUTER_PROMPT
from src.llm_client import get_llm
from src.query_transformation.rewriting import rewrite_query
from src.query_transformation.multi_query import generate_multi_queries
from src.query_transformation.decomposition import decompose_query
from src.query_transformation.hyde import generate_hyde_document
from src.retrieval.retriever import retrieve_single, retrieve_multi
from src.reranking.reranker import rerank_chunks
from src.generation.generator import generate_answer
from src import config


class PipelineState(TypedDict):
    query: str
    technique: str
    reason: str
    retrieved: List[dict]
    reranked: List[dict]
    answer: str
    input_tokens: int
    output_tokens: int


def classify_node(state: PipelineState) -> PipelineState:
    llm = get_llm().with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
    return {**state, "technique": decision.technique, "reason": decision.reason}


def route_after_classify(state: PipelineState) -> str:
    return state["technique"]  # matches node names below exactly


def none_node(state: PipelineState) -> PipelineState:
    chunks = retrieve_single(state["query"], top_k=config.RETRIEVAL_TOP_K_SINGLE, input_type="search_query")
    return {**state, "retrieved": chunks}


def rewriting_node(state: PipelineState) -> PipelineState:
    rewritten = rewrite_query(state["query"])
    chunks = retrieve_single(rewritten, top_k=config.RETRIEVAL_TOP_K_SINGLE, input_type="search_query")
    return {**state, "retrieved": chunks}


def multi_query_node(state: PipelineState) -> PipelineState:
    variants = generate_multi_queries(state["query"])
    chunks = retrieve_multi(variants, top_k_per_query=config.RETRIEVAL_TOP_K_PER_SUBQUERY)
    return {**state, "retrieved": chunks}


def decomposition_node(state: PipelineState) -> PipelineState:
    sub_questions = decompose_query(state["query"])
    chunks = retrieve_multi(sub_questions, top_k_per_query=config.RETRIEVAL_TOP_K_PER_SUBQUERY)
    return {**state, "retrieved": chunks}


def hyde_node(state: PipelineState) -> PipelineState:
    hyde_doc = generate_hyde_document(state["query"])
    # embedded as 'search_document', not 'search_query' — see reasoning in retriever.py
    chunks = retrieve_single(hyde_doc, top_k=config.RETRIEVAL_TOP_K_SINGLE, input_type="search_document")
    return {**state, "retrieved": chunks}


def rerank_node(state: PipelineState) -> PipelineState:
    # always reranked against the ORIGINAL query, never the transformed one
    reranked = rerank_chunks(state["query"], state["retrieved"], top_n=config.RERANK_TOP_N)
    return {**state, "reranked": reranked}


def generate_node(state: PipelineState) -> PipelineState:
    result = generate_answer(state["query"], state["reranked"])
    return {
        **state,
        "answer": result["answer"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("classify", classify_node)
    graph.add_node("none", none_node)
    graph.add_node("rewriting", rewriting_node)
    graph.add_node("multi_query", multi_query_node)
    graph.add_node("decomposition", decomposition_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_after_classify, {
        "none": "none",
        "rewriting": "rewriting",
        "multi_query": "multi_query",
        "decomposition": "decomposition",
        "hyde": "hyde",
    })

    for technique_node in ["none", "rewriting", "multi_query", "decomposition", "hyde"]:
        graph.add_edge(technique_node, "rerank")

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()