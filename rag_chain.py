"""
STEP 5, 6, 7: Load the vector store, set up an open-source LLM, wire together
a grounded RAG chain, and add guardrails against hallucination.

Two LLM backends are supported — pick whichever works better for you:
  A) Hugging Face Inference API  (pure HF, but can be rate-limited on free tier)
  B) Groq free tier              (serves the SAME open-source Llama 3 model, much faster)
Both are open-source models — Groq is just a faster host for step 5's model.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity score threshold below which we refuse to answer (guardrail, Step 7)
# Lower distance = more similar for FAISS's default L2 metric.
RELEVANCE_DISTANCE_THRESHOLD = 1.0

SYSTEM_PROMPT = """You are a helpful banking customer support assistant.

STRICT RULES:
1. Answer ONLY using the information in the provided context below.
2. If the context does not contain enough information to answer the question,
   respond exactly with: "I don't have that information in my knowledge base.
   Please contact customer care or visit your nearest branch for this query."
3. Do NOT make up policies, numbers, interest rates, or procedures that are not
   explicitly stated in the context.
4. Keep answers concise and cite the FAQ category when relevant.

Context:
{context}

Question: {question}

Answer:"""


def load_llm(backend: str = "groq"):
    """STEP 5: Load an open-source LLM via your chosen backend."""
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
               model="openai/gpt-oss-20b",  # open-source Llama 3.1, hosted fast on Groq
            temperature=0.1,               # low temperature = fewer hallucinations
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif backend == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.1,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        return ChatHuggingFace(llm=llm)
    else:
        raise ValueError("backend must be 'groq' or 'huggingface'")


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('category', 'General')}] {d.page_content}" for d in docs)


def check_relevance(vectorstore, query: str, k: int = 4):
    """
    STEP 7 (guardrail): Run similarity search WITH scores first. If the best
    match is too dissimilar, short-circuit and refuse rather than letting the
    LLM improvise an answer from weak context.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return [], True  # no docs at all -> out of scope

    best_distance = results[0][1]
    out_of_scope = best_distance > RELEVANCE_DISTANCE_THRESHOLD
    docs = [doc for doc, _score in results]
    return docs, out_of_scope


def build_chain(backend: str = "groq"):
    llm = load_llm(backend)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain


def answer_question(query: str, vectorstore, chain) -> dict:
    """STEP 6 + 7 combined: retrieve, guardrail-check, then generate."""
    docs, out_of_scope = check_relevance(vectorstore, query)

    if out_of_scope:
        return {
            "answer": "I don't have that information in my knowledge base. "
                      "Please contact customer care or visit your nearest branch for this query.",
            "sources": [],
        }

    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": query})
    sources = list({d.metadata.get("category", "General") for d in docs})
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Quick manual test — run: python rag_chain.py
    vectorstore = load_retriever()
    chain = build_chain(backend="groq")

    test_questions = [
        "How do I open a savings account?",
        "What is the interest rate on a home loan?",
        "What's the weather like today?",  # should be refused — out of scope
    ]

    for q in test_questions:
        result = answer_question(q, vectorstore, chain)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
