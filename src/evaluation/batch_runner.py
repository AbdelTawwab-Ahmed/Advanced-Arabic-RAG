import asyncio
import time
from typing import List, TypedDict
from src.routing.pipeline import build_pipeline
from src.evaluation.evaluator import evaluate_answer
from src import config


PIPELINE = build_pipeline()


class EvalQuestion(TypedDict):
    question: str
    ground_truth: str


class ResultRow(TypedDict):
    question: str
    technique: str
    answer: str
    ground_truth: str
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    correctness: float
    input_tokens: int
    output_tokens: int
    total_cost_usd: float


async def run_one(item: EvalQuestion) -> ResultRow:
    result = PIPELINE.invoke({"query": item["question"]})

    contexts = [c["text"] for c in result["reranked"]]

    scores = await evaluate_answer(
        question=item["question"],
        answer=result["answer"],
        contexts=contexts,
        ground_truth=item["ground_truth"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )

    return {
        "question": item["question"],
        "technique": result["technique"],
        "answer": result["answer"],
        "ground_truth": item["ground_truth"],
        **scores,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def run_batch(items: List[EvalQuestion], delay_seconds: float = config.EVAL_BATCH_DELAY_SECONDS) -> List[ResultRow]:
    rows = []
    for i, item in enumerate(items):
        print(f"\n[{i+1}/{len(items)}] {item['question'][:50]}...\n")
        row = asyncio.run(run_one(item))
        rows.append(row)
        if i < len(items) - 1:
            time.sleep(delay_seconds)
    return rows