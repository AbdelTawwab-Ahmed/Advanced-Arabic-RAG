import os
from src.query_transformation.multi_query import generate_multi_queries

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_query = "ما هي إجراءات إصدار البطاقات؟"

variants = generate_multi_queries(test_query)

output_path = os.path.join(OUTPUT_DIR, "test_multi_query_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Original: {test_query}\n\n")
    for i, v in enumerate(variants, 1):
        f.write(f"Variant {i}: {v}\n")

print(f"\nDone — results written to {output_path}")