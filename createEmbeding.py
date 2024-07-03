# Import necessary modules and classes from LangChain library
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

# Path to the local PDF file to be loaded
local_path = "./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf"

# Load the PDF file using PyPDFLoader
loader = PyPDFLoader(file_path=local_path)
data = loader.load()  # Load the entire PDF document into memory
first_chapter = data[10:32]  # Extract a subset of pages (from 11th to 32nd page)

# Split the extracted document into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, separators=["\n", "\n\n"])
text_chunks = text_splitter.split_documents(first_chapter)

# Create a vector store (Chroma) from the text chunks using Ollama embeddings
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag",
    persist_directory = './chroma',
)
vector_store.persist()  # Persist the vector store to disk