# 📊 Financial Document Analyzer

A RAG (Retrieval Augmented Generation) system that lets you upload any financial PDF and ask questions about it in plain English. Built from scratch — no LangChain.

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/SachinK011/financial-document-analyzer)**

---

## What it does

Upload a bank statement, loan agreement, or financial report and ask questions like:

- *"What is the average monthly surplus?"*
- *"How much is paid for rent each month?"*
- *"What are the recurring investment expenses?"*
- *"What is the closing balance?"*
- *"What is the total monthly income including freelance?"*

The app retrieves the most relevant sections from your document and answers strictly from that context, which significantly reduces the chance of the model making up information.

---

## Architecture

```
PDF Upload → Text Extraction (PyMuPDF)
          → Chunking (500 chars, 50 overlap)
          → Embeddings (HuggingFace all-MiniLM-L6-v2)
          → Vector Store (FAISS)
          → Retrieval (top-6 chunks)
          → LLM Answer (Groq Llama 3.1)
```

No LangChain — every component is built and wired manually so the pipeline is fully transparent and explainable.

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyMuPDF |
| Text Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| LLM | Groq API — Llama 3.1 8B Instant |
| Frontend | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
financial-document-analyzer/
├── app.py                      # Streamlit frontend + RAG pipeline
├── requirements.txt            # Dependencies
├── sample_bank_statement.pdf   # Test document (fictional data)
├── 01_rag_pipeline.ipynb       # Full pipeline built step by step
├── 02_evaluation.ipynb         # Answer quality evaluation
└── .env.example                # API key template
```

---

## Run Locally

```bash
git clone https://github.com/Sachin-kalaiselvan/financial-document-analyzer
cd financial-document-analyzer
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## Evaluation Results

Tested on a 3-page fictional bank statement (Oct–Dec 2024):

| Question | Result |
|---|---|
| Average monthly surplus | ✅ Correct |
| Rent per month | ✅ Correct |
| Closing balance | ✅ Correct |
| Total monthly income | ✅ Correct |
| Recurring investments | ✅ Correct |
| Opening balance | ✅ Correct |
| Surplus % of income | ✅ Correct |
| Questions not in document | ✅ Correctly refused |

**Accuracy: 8/8 factual questions answered correctly**

---

## Key Learnings

- Chunk size of 500 with overlap 50 works well for tabular financial data
- Increasing retrieval k from 4 to 6 improved coverage for edge case questions
- Question phrasing affects retrieval — shorter, direct questions work best
- Grounding the LLM strictly to retrieved context eliminates hallucination

---

## Author

**Sachin K** — [GitHub](https://github.com/Sachin-kalaiselvan)
