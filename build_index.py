import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_FILE = "data/financebench.json"
INDEX_FILE = "vectorstore/index.faiss"
METADATA_FILE = "vectorstore/metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading FinanceBench evidence...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

texts = [record["text"] for record in records]

print(f"Loaded {len(texts)} evidence chunks.")

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print("Creating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

Path("vectorstore").mkdir(exist_ok=True)

faiss.write_index(index, INDEX_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print()
print("====================================")
print("FAISS index created successfully!")
print("====================================")
print(f"Chunks indexed: {len(records)}")
print(f"Embedding dimension: {dimension}")
print(f"Index: {INDEX_FILE}")
print(f"Metadata: {METADATA_FILE}")