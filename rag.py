from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
import os


load_dotenv()

CHUNK_SIZE = 500
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR = Path(__file__).parent / "resources/vector_store"
COLLECTION_NAME = "real_estate"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure token is set
if not hf_token:
    raise ValueError("Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN in .env or system environment.")

# Authenticate before using the model
login(token=hf_token)

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR)
        )


def process_documents(file_paths):
    """
    Loads documents from local files (PDF, DOCX, or TXT) and stores
    their chunks in the vector database.

    :param file_paths: list of local file paths to ingest
    """

    yield "Initializing components...✅"
    initialize_components()
    vector_store.reset_collection()

    yield "Resetting vector store...✅"

    # ── CHANGED ──────────────────────────────────────────────────────────────
    # Previously: a single UnstructuredURLLoader(urls=urls).load() call.
    # Now: iterate over each file, pick the right loader by extension, and
    # accumulate all Document objects into `data`.
    #
    # Loader choice:
    #   .pdf  → PyPDFLoader   : splits by page; each page becomes one Document
    #   .docx → Docx2txtLoader: reads the full docx as one Document
    #   .txt  → TextLoader    : reads the full text file as one Document
    #
    # All three return List[Document] with .page_content and .metadata, which
    # is exactly what UnstructuredURLLoader returned — so nothing below changes.
    data = []
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            # Skip unsupported formats gracefully
            yield f"⚠️ Skipping unsupported file type: {Path(file_path).name}"
            continue

        data.extend(loader.load())

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )

    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector Database is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

    result = chain.invoke({"question": query}, return_only_outputs=True)

    sources = result.get("sources", "")

    return result['answer'], sources