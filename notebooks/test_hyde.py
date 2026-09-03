import os
from src.query_transformation.hyde import generate_hyde_document

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_query = "كيف تضمن وحدة الانذار المركزي استمرارية الأمان في البنك؟"

hyde_doc = generate_hyde_document(test_query)

output_path = os.path.join(OUTPUT_DIR, "test_hyde_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Original query: {test_query}\n\n")
    f.write(f"Hypothetical document:\n{hyde_doc}\n")

print(f"\nDone — results written to {output_path}")