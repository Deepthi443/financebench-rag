# FinanceBench Hallucination-Proof Document Q&A Bot

A RAG-based financial document question-answering system built using FinanceBench.

## Tech Stack
- Python
- Sentence Transformers
- FAISS
- Groq
- Streamlit

## Features
- Document evidence retrieval
- Semantic search using FAISS
- Source document, page and chunk citations
- Hallucination guardrail
- Exact refusal when evidence is insufficient

## Example

Question:
What is the FY2018 capital expenditure amount (in USD millions) for 3M?

Answer:
$1577.00

Source:
3M_2018_10K
