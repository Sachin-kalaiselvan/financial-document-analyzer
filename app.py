import streamlit as st
import fitz
import faiss
import numpy as np
import groq
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Document Analyzer")
st.caption("Upload any financial PDF and ask questions in plain English.")
st.divider()

groq_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
st.sidebar.caption("Get your free key at console.groq.com")
st.sidebar.divider()
st.sidebar.markdown("**Try asking:**")
st.sidebar.markdown("- What is the average monthly surplus?")
st.sidebar.markdown("- How much is paid for rent?")
st.sidebar.markdown("- What are the recurring investments?")
st.sidebar.markdown("- What is the closing balance?")
st.sidebar.markdown("- What is the total monthly income?")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        text += f"\n--- Page {i+1} ---\n{page.get_text()}"
    doc.close()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, chunks, k=6):
    vec = embedder.encode([query])
    _, indices = index.search(np.array(vec), k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_llm(question, context_chunks, api_key):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a financial document assistant.
Answer using ONLY the document excerpt below.
If the answer is not in the excerpt, say "I could not find that in the document."
Be concise and use numbers where available.

--- DOCUMENT EXCERPT ---
{context}
--- END ---

Question: {question}
Answer:"""
    client = groq.Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip(), context_chunks

uploaded_file = st.file_uploader("Upload a financial PDF", type="pdf")

if uploaded_file:
    file_key = uploaded_file.name

    if "indexed_file" not in st.session_state or st.session_state.indexed_file != file_key:
        with st.spinner("Reading and indexing your document..."):
            pdf_bytes = uploaded_file.read()
            raw_text = extract_text(pdf_bytes)
            chunks = chunk_text(raw_text)
            index = build_index(chunks)
            st.session_state.indexed_file = file_key
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.chunk_count = len(chunks)

    st.success(f"Document indexed — {st.session_state.chunk_count} chunks ready.")
    st.divider()

    question = st.text_input("Ask a question about your document")

    if question:
        if not groq_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                answer, sources = ask_llm(
                    question,
                    retrieve(question, st.session_state.index, st.session_state.chunks),
                    groq_key
                )
            st.markdown("### Answer")
            st.success(answer)
            with st.expander("View source chunks used to answer"):
                for i, chunk in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk)
            st.divider()
