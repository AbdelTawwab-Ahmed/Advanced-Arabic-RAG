import os
from src.query_transformation.decomposition import decompose_query

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_query = "ما هي خطوات إصدار البطاقة وما هي خطوات إلغائها؟"

sub_questions = decompose_query(test_query)

output_path = os.path.join(OUTPUT_DIR, "test_decomposition_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Original: {test_query}\n\n")
    for i, sq in enumerate(sub_questions, 1):
        f.write(f"Sub-question {i}: {sq}\n")

print(f"\nDone — results written to {output_path}")