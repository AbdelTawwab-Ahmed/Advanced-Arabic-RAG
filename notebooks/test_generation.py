import os
from src.retrieval.retriever import retrieve_single
from src.reranking.reranker import rerank_chunks
from src.generation.generator import generate_answer

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

query = "ما هي خطوات إصدار البطاقة؟"
retrieved = retrieve_single(query, top_k=10)
reranked = rerank_chunks(query, retrieved, top_n=5)
result = generate_answer(query, reranked)

output_path = os.path.join(OUTPUT_DIR, "test_generation_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Query: {query}\n\n")
    f.write(f"Answer:\n{result['answer']}\n\n")
    f.write(f"Input tokens: {result['input_tokens']}\n")
    f.write(f"Output tokens: {result['output_tokens']}\n")

print(f"\nDone — results written to {output_path}")