import os
import fitz  
import faiss
import pickle
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

PDF_FOLDER = "data"
VECTOR_FOLDER = "vector_store"
INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_FOLDER, "chunks.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3"


# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

model = load_embedding_model()


# PDF TEXT EXTRACTION

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    doc.close()
    return text

# TEXT CHUNKING

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# LOAD ALL PDF DOCUMENTS

def load_documents():
    all_chunks = []

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        text = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "source": pdf,
                "text": chunk
            })

    return all_chunks

# BUILD VECTOR DATABASE

def create_vector_store():
    chunks_data = load_documents()

    if len(chunks_data) == 0:
        return False

    texts = [item["text"] for item in chunks_data]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks_data, f)

    return True

def load_vector_store():
    if not os.path.exists(INDEX_FILE):
        return None, None

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        chunks_data = pickle.load(f)

    return index, chunks_data

# RETRIEVE RELEVANT DOCUMENTS

def retrieve_context(query, top_k=3):
    index, chunks_data = load_vector_store()

    if index is None:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    retrieved = []

    for idx in indices[0]:
        if idx < len(chunks_data):
            retrieved.append(chunks_data[idx])

    return retrieved

#STREAMLIT USER INTERFACE

def generate_answer(question, contexts):

    context_text = "\n\n".join(
        [f"Source: {c['source']}\n{c['text']}" for c in contexts]
    )

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer the user's question ONLY using the context provided below.
If the answer cannot be found in the context, reply:
"I could not find that information in the provided documents."

Context:
{context_text}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        return response.json()["response"]

    except Exception as e:
        return f"Error connecting to Ollama: {e}"


st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Enterprise Knowledge Assistant")
st.write("### Retrieval-Augmented Generation (RAG)")

st.sidebar.header("Document Processing")

if st.sidebar.button("Build / Rebuild Knowledge Base"):
    with st.spinner("Reading PDFs and building vector database..."):
        success = create_vector_store()

    if success:
        st.sidebar.success("Knowledge base built successfully!")
    else:
        st.sidebar.error(
            "No PDF files found. Please place PDF files inside the 'data' folder."
        )

st.markdown("---")

question = st.text_input(
    "Ask a question about your enterprise documents:"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents..."):
            retrieved_context = retrieve_context(question)

        if len(retrieved_context) == 0:
            st.error(
                "Knowledge base not found. Please build the vector database first."
            )

        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    question,
                    retrieved_context
                )

            st.subheader("Answer")
            st.write(answer)

            st.markdown("---")
            st.subheader("Retrieved Context")

            for i, item in enumerate(retrieved_context, start=1):
                with st.expander(
                    f"Context {i} - Source: {item['source']}"
                ):
                    st.write(item["text"])

