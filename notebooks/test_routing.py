import os
from src.routing.router import build_router_graph

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = build_router_graph()

test_queries = [
    "ما هي مدة الاحتفاظ بتسجيلات المراقبة التلفزيونية؟",
    "بدي اعرف شغلة الكرت والبصمة كيف بتنعمل",
    "ما هي خطوات إصدار البطاقة وما هي خطوات إلغائها؟",
    "كيف تضمن وحدة الانذار المركزي استمرارية الأمان في البنك؟",
]

output_path = os.path.join(OUTPUT_DIR, "test_router_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for q in test_queries:
        result = router.invoke({"query": q, "technique": None, "reason": None})
        f.write(f"Query: {q}\n")
        f.write(f"-> {result['technique']} | {result['reason']}\n\n")

print(f"Done — results written to {output_path}")