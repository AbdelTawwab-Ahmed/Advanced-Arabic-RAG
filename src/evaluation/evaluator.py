import os
from typing import List, TypedDict
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness, ResponseRelevancy, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_cohere import CohereEmbeddings
from src.llm_client import get_eval_llm
from src import config


def get_ragas_llm():
    return LangchainLLMWrapper(get_eval_llm())


def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(CohereEmbeddings(model=config.COHERE_EMBED_MODEL, cohere_api_key=config.COHERE_API_KEY))


class EvalResult(TypedDict):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    correctness: float
    total_cost_usd: float


async def evaluate_answer(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    input_tokens: int,
    output_tokens: int,
) -> EvalResult:
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    context_precision = LLMContextPrecisionWithoutReference(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1)
    factual_correctness = FactualCorrectness(llm=llm, mode="recall")

    cost = (input_tokens / 1_000_000 * config.GEMINI_PRICE_PER_1M_INPUT) + \
           (output_tokens / 1_000_000 * config.GEMINI_PRICE_PER_1M_OUTPUT)

    return {
        "context_relevance": await context_precision.single_turn_ascore(sample),
        "faithfulness": await faithfulness.single_turn_ascore(sample),
        "answer_relevance": await answer_relevancy.single_turn_ascore(sample),
        "correctness": await factual_correctness.single_turn_ascore(sample),
        "total_cost_usd": round(cost, 6),
    }