# 🏦 Banking FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers banking questions
(loans, cards, KYC, account opening, online banking) grounded strictly in a
knowledge base — with a guardrail that makes it refuse to answer instead of
hallucinating when a question is out of scope.

## Architecture

```
User question
     │
     ▼
[HuggingFace embedding: all-MiniLM-L6-v2]  (local, free)
     │
     ▼
[FAISS similarity search]  ──► relevance check (guardrail)
     │                              │
     │                     too dissimilar? → refuse gracefully
     ▼
[Top-k relevant FAQ chunks as context]
     │
     ▼
[Llama 3.1 8B via Groq]  (open-source model, fast free-tier inference)
     │
     ▼
Grounded answer + source category
```

## Stack
- **Orchestration:** LangChain (LCEL)
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
- **Vector store:** FAISS
- **LLM:** Llama 3.1 8B (open-source) — served via Groq's free tier for speed;
  swappable for Hugging Face's Inference API (`backend="huggingface"` in `rag_chain.py`)
- **UI:** Streamlit
- **Data:** Banking FAQs (loans, cards, KYC, deposits, online banking, disputes)

## Setup

1. **Clone & install**
   ```bash
   git clone <your-repo-url>
   cd banking-rag-chatbot
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get API keys**
   - Hugging Face token: https://huggingface.co/settings/tokens
   - Groq API key (free): https://console.groq.com/keys
   - Copy `.env.example` to `.env` and fill in your keys.

3. **(Optional) Expand the dataset**
   The seed dataset (`data/bank_faqs.csv`) has 20 Q&A pairs across 8 categories.
   To make this a stronger resume project, grow it to 150-300+ pairs:
   - Edit `scripts/scrape_faqs.py` with real bank FAQ page URLs + CSS selectors, or
   - Download the Kaggle "Bank FAQ Dataset" and merge it into `data/bank_faqs.csv`

4. **Build the vector store**
   ```bash
   python scripts/build_vectorstore.py
   ```

5. **Test the RAG chain from the command line**
   ```bash
   python rag_chain.py
   ```

6. **Run the chat UI**
   ```bash
   streamlit run app.py
   ```

## Guardrail against hallucination
Before generating an answer, the chatbot runs a similarity search and checks
the distance score of the best match. If nothing in the knowledge base is
close enough to the question, it responds with a fixed fallback message
instead of asking the LLM to improvise — this is the key design choice that
separates this from a toy chatbot and is worth highlighting in interviews.

## Deployment
Free options:
- **Streamlit Community Cloud** (share.streamlit.io) — connect your GitHub repo directly
- **Hugging Face Spaces** (Streamlit SDK) — nice touch since the project already
  uses Hugging Face for embeddings

## Possible extensions (good "future work" talking points)
- Add PDF ingestion for full bank policy documents, not just FAQs
- Add conversation memory (multi-turn context)
- Add a feedback loop (thumbs up/down) to track retrieval quality
- Evaluate with RAGAS (faithfulness, answer relevance metrics)
