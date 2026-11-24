# financial-document-analyzer

upload a financial pdf, ask it questions in normal english. RAG built from scratch, no langchain.

**demo:** https://huggingface.co/spaces/SachinK011/financial-document-analyzer

## what you can ask

throw in a bank statement, loan agreement, annual report, then stuff like:

- *"what's the average monthly surplus?"*
- *"how much goes to rent each month?"*
- *"what are the recurring investment expenses?"*
- *"closing balance?"*
- *"total monthly income incl. freelance?"*

it pulls the most relevant chunks of your doc and answers *only* from those, which keeps the model from making stuff up.

## pipeline

```
pdf → text (pymupdf)
    → chunks (500 chars, 50 overlap)
    → embeddings (all-MiniLM-L6-v2)
    → FAISS
    → top 6 chunks
    → answer (groq, llama 3.1 8b instant)
```

no langchain on purpose. every piece is written out so it's easy to follow and swap. streamlit frontend, hosted on HF spaces.

```
app.py                     streamlit + the pipeline
01_rag_pipeline.ipynb      built step by step
02_evaluation.ipynb        answer quality eval
sample_bank_statement.pdf  fake test doc
.env.example
```

## run it

```bash
git clone https://github.com/Sachin-kalaiselvan/financial-document-analyzer
cd financial-document-analyzer
pip install -r requirements.txt
streamlit run app.py
```

paste a groq key in the sidebar when it asks (free at [console.groq.com](https://console.groq.com)).

## eval

3-page fake bank statement (oct-dec 2024): avg monthly surplus, rent, closing balance, total income, recurring investments, opening balance, surplus as % of income, all correct. questions that aren't in the doc get refused like they should. **8/8.**
