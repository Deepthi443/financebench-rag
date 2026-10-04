from datasets import load_dataset
import json
from pathlib import Path

print("Loading FinanceBench dataset...")

ds = load_dataset(
    "PatronusAI/financebench",
    split="train"
)

records = []
chunk_id = 0

for row in ds:
    for evidence in row["evidence"]:

        text = evidence["evidence_text"]

        if not text or not text.strip():
            continue

        records.append({
            "chunk_id": chunk_id,
            "financebench_id": row["financebench_id"],
            "company": row["company"],
            "doc_name": evidence["doc_name"],
            "page": evidence["evidence_page_num"],
            "text": text.strip(),
            "question": row["question"],
            "answer": row["answer"]
        })

        chunk_id += 1

Path("data").mkdir(exist_ok=True)

with open("data/financebench.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print()
print("====================================")
print("FinanceBench preparation completed!")
print("====================================")
print(f"Evidence chunks saved: {len(records)}")
print("Output file: data/financebench.json")