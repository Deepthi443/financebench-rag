import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "vectorstore/index.faiss"
METADATA_FILE = "vectorstore/metadata.json"


class Retriever:

    def __init__(self, top_k=5):

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_FILE)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.top_k = top_k

    def retrieve(self, question):

        # Search the whole small corpus
        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            len(self.metadata)
        )

        question_lower = question.lower()

        # Detect year
        year_match = re.search(r"\b(20\d{2})\b", question_lower)
        question_year = year_match.group(1) if year_match else None

        # Detect company
        companies = set(
            item.get("company", "").lower()
            for item in self.metadata
            if item.get("company")
        )

        question_company = None

        for company in companies:
            if company and company in question_lower:
                question_company = company
                break

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            semantic_score = float(score)
            boost = 0

            document = result.get("doc_name", "").lower()
            company = result.get("company", "").lower()

            # Strong company match
            if question_company and company == question_company:
                boost += 0.35

            # Strong year match
            if question_year and question_year in document:
                boost += 0.45

            result["original_score"] = semantic_score
            result["score"] = semantic_score + boost

            results.append(result)

        # Highest combined score first
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:self.top_k]


if __name__ == "__main__":

    retriever = Retriever()

    question = input("Enter your question: ")

    results = retriever.retrieve(question)

    print("\n===== RETRIEVED EVIDENCE =====\n")

    for result in results:

        print(f"Document: {result['doc_name']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Original similarity: {result['original_score']:.4f}")
        print(f"Text: {result['text'][:500]}")

        print("\n" + "-" * 60 + "\n")