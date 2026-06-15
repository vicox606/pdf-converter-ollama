from pypdf import PdfReader
import sys
import os

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompt import PromptTemplate
from langchain.chains import RetrievalQA 

class SimpleDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def _make_doc(page_content, page_number):
    return SimpleDoc(page_content, {"page": page_number})


def load_pdf(path):
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(_make_doc(text, i + 1))
    return docs


# Determine PDF path (CLI arg overrides default uploaded file)
default_pdf = "RAG problem statement.pdf"
pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

# Load the PDF document
documents = load_pdf(pdf_path)

print(f"Loaded {len(documents)} pages")


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    chunks = []
    for doc in documents:
        text = doc.page_content or ""
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + chunk_size)
            chunk_text = text[start:end]
            metadata = dict(doc.metadata) if getattr(doc, 'metadata', None) else {}
            metadata.update({"chunk_start": start, "chunk_end": end})
            chunks.append(SimpleDoc(chunk_text, metadata))
            if end == L:
                break
            start = end - chunk_overlap
            if start < 0:
                start = 0
    return chunks


docs = split_documents(documents, chunk_size=500, chunk_overlap=100)
print(f"Split into {len(docs)} chunks")

# Create embeddings using sentence-transformers
st_model = SentenceTransformer("sentence-transformers/all-miniLM-L6-v2")

def embedding_fn(texts):
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True
    vectors = st_model.encode(texts, show_progress_bar=False)
    return vectors if not single else vectors[0]

# Store it ChromaDB

vectorestore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_fn,
    persist_directory="./chroma_db",
)

retriever = vectorestore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Load the Ollama LLM

llm = OllamaLLM(model="llama3.2")

# Custom prompt template
prompt_template = """
Your are an Enterprise Knowledge Assistant.

Use Only The provided context to answer the question.
If the answer is not available in the context, Say:

"I could not find that information in the enterprise document."

Context:{context}

Question:{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Build RAG Chain 

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt
    },
    return_source_documents=True
)

# Chat Loop

print("\n=== Enterprise RAG Assistant ===")
print("Type 'exit' to Quit.\n")

while True:
    query = input("What Would YOU like Know: ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})

    print("\nAnswer:")
    print(result["result"])

    print("\nSources used: ")
    for doc in result["sources_documents"]:
        print(f"page {doc.metadata.get('page','N/A')}")
        print("-"*50)

