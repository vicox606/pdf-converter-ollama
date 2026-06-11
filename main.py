from langchain_community.document_loaders import PyPDFLoader
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

# Load the PDF document
loader = PyPDFLoader(".pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLM-L6-v2")

# Store it ChromaDB

vectorestore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
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
prompt = prompt_template(
    template=template,
    input_variable=["context","question"]
)

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
    


