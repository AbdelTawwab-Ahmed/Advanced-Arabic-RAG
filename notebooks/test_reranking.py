import os
from src.retrieval.retriever import retrieve_single
from src.reranking.reranker import rerank_chunks

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

query = "ما هي مدة الاحتفاظ بتسجيلات المراقبة التلفزيونية؟"
retrieved = retrieve_single(query, top_k=10)
reranked = rerank_chunks(query, retrieved, top_n=5)

output_path = os.path.join(OUTPUT_DIR, "test_reranking_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== Before reranking (hybrid RRF order) ===\n")
    for c in retrieved:
        f.write(f"score={c['score']:.4f} | page {c['page_number']}\n")

    f.write("\n=== After reranking ===\n")
    for c in reranked:
        f.write(f"rerank_score={c['rerank_score']:.4f} | page {c['page_number']}\n")

print(f"\nDone — results written to {output_path}")