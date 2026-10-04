from retriever import Retriever
from generator import Generator


class RAGSystem:

    def __init__(self):
        self.retriever = Retriever(top_k=5)
        self.generator = Generator()

    def ask(self, question):

        results = self.retriever.retrieve(question)

        if not results:
            return "I don't know based on the provided documents"

        # Basic evidence check
        if results[0]["score"] < 0.50:
            return "I don't know based on the provided documents"

        answer = self.generator.generate(
            question,
            results
        )

        return answer


if __name__ == "__main__":

    rag = RAGSystem()

    question = input("Ask your question: ")

    answer = rag.ask(question)

    print("\n===== ANSWER =====\n")
    print(answer)