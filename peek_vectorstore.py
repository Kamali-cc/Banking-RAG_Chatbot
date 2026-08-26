"""
Just for curiosity: loads the FAISS index you built and runs one sample
search, so you can SEE that it's actually retrieving relevant FAQ chunks
before we wire up the LLM. This doesn't need any API keys.

Usage: python peek_vectorstore.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model (should be fast, already cached from last run)...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)

query = "How do I get a new debit card if mine is lost?"
print(f"\nSearching for: '{query}'\n")

results = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Match {i} (distance score: {score:.4f}, lower = more similar) ---")
    print(doc.page_content)
    print(f"Category: {doc.metadata.get('category')}")
    print()
