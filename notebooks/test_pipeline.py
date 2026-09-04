import os
from src.routing.pipeline import build_pipeline

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = build_pipeline()

test_queries = [
    "ما هي مدة الاحتفاظ بتسجيلات المراقبة التلفزيونية؟",
    "بدي اعرف شغلة الكرت والبصمة كيف بتنعمل",
]

output_path = os.path.join(OUTPUT_DIR, "test_pipeline_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for q in test_queries:
        result = pipeline.invoke({"query": q})
        f.write(f"Query: {q}\n")
        f.write(f"Technique: {result['technique']} ({result['reason']})\n")
        f.write(f"Answer:\n{result['answer']}\n")
        f.write(f"Tokens: in={result['input_tokens']} out={result['output_tokens']}\n")
        f.write("\n" + "="*60 + "\n\n")

print(f"\nDone — results written to {output_path}")