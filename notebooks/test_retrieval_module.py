import os
from src.retrieval.retriever import retrieve_single, retrieve_multi

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "test_retrieval_module_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== retrieve_single ===\n")
    for c in retrieve_single("ما هي مدة الاحتفاظ بتسجيلات المراقبة التلفزيونية؟", top_k=3):
        f.write(f"score={c['score']:.4f} | {c['source_pdf']} | page {c['page_number']}\n")

    f.write("\n=== retrieve_multi ===\n")
    variants = [
        "ما هي خطوات إصدار البطاقة؟",
        "كيف يتم إصدار البطاقة؟",
    ]
    for c in retrieve_multi(variants, top_k_per_query=5):
        f.write(f"score={c['score']:.4f} | {c['source_pdf']} | page {c['page_number']}\n")

print(f"\nDone — results written to {output_path}")