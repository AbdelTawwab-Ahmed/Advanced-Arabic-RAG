import os
from src.query_transformation.rewriting import rewrite_query

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_queries = [
    "بدي اعرف شغلة الكرت والبصمة كيف بتنعمل",
    "شو القصة مع كاميرات المراقبة يعني لحتى إمتى بتضل مسجلة",
]

output_path = os.path.join(OUTPUT_DIR, "test_rewriting_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for q in test_queries:
        rewritten = rewrite_query(q)
        f.write(f"Original:  {q}\n")
        f.write(f"Rewritten: {rewritten}\n\n")

print(f"\nDone — results written to {output_path}")