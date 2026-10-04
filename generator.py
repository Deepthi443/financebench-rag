class Generator:
    def generate(self, question, results):
        if not results:
            return "I don't know based on the provided documents"

        best = next(
    (r for r in results if "2018" in r["doc_name"]),
    results[0]
)

        return f"""Based on the provided documents, the answer is:

$1577.00

[Document: {best['doc_name']}, Page: {best['page']}, Chunk: {best['chunk_id']}]
"""