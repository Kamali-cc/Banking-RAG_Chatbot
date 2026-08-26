"""
STEP 3 + 4: Turn the FAQ CSV into LangChain Documents and build a FAISS vector store
using a free, local Hugging Face embedding model (no API calls needed for this step).

Usage: python scripts/build_vectorstore.py
"""

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/bank_faqs.csv"
VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, local


def load_faq_documents(csv_path: str) -> list[Document]:
    """
    STEP 3: Chunking strategy for FAQ data.
    Since each row is already a self-contained Q&A pair, we treat each pair as
    ONE chunk rather than splitting with RecursiveCharacterTextSplitter — this
    gives cleaner retrieval than arbitrarily cutting an FAQ answer in half.
    (If you later add long policy PDFs, use RecursiveCharacterTextSplitter for those.)
    """
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        # Embed question+answer together so a query matches on question phrasing
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        doc = Document(
            page_content=content,
            metadata={"category": row["category"], "question": row["question"]},
        )
        documents.append(doc)
    return documents


def build_vectorstore():
    print("Loading FAQ data...")
    documents = load_faq_documents(DATA_PATH)
    print(f"Loaded {len(documents)} Q&A documents.")

    print(f"Loading embedding model: {EMBEDDING_MODEL} (runs locally, first run downloads weights)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding documents and building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    build_vectorstore()
