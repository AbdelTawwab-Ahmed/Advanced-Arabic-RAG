import os
import csv
from src.evaluation.batch_runner import run_batch

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_items = [
    {
        "question": "ما هي خطوات إصدار البطاقة؟",
        "ground_truth": "خطوات إصدار البطاقة هي: استلام الطلب عبر البريد الرسمي، التحقق من بيانات الموظف، إرسال الطلب لمدير مساعد الانذار المركزي، التأكد من اكتمال المعلومات، الموافقة على الطلب، استلام موافقة صاحب الصلاحية، ثم إرسال الموافقة لإصدار البطاقة وتفعيلها ومنحها الصلاحيات.",
    },
]

rows = run_batch(test_items)

output_path = os.path.join(OUTPUT_DIR, "test_batch_results.csv")
with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Done — {len(rows)} row(s) written to {output_path}")