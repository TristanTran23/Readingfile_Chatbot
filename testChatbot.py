# Import necessary modules and classes from LangChain library
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Path to the local PDF file to be loaded
local_path = "./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf"

# Load the PDF file using PyPDFLoader
loader = PyPDFLoader(file_path=local_path)
data = loader.load()  # Load the entire PDF document into memory
first_chapter = data[11:32]  # Extract a subset of pages (from 12th to 32nd page)

# Split the extracted document into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, separators=["\n", "\n\n"])
text_chunks = text_splitter.split_documents(first_chapter)

# Create a vector store (Chroma) from the text chunks using Ollama embeddings
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)

# Initialize a ChatOllama language model
local_model = "llama3"
llm = ChatOllama(model=local_model)

# Define a prompt template for generating alternative versions of user queries
QUERY_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Create a multi-query retriever from the vector store and ChatOllama model with the defined prompt
retriever = MultiQueryRetriever.from_llm(
    vector_store.as_retriever(),
    llm,
    prompt=QUERY_PROMPT,
)

# Define a template for the prompt response
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Initialize a ChatPromptTemplate from the defined template
prompt = ChatPromptTemplate.from_template(template)

# Define a chain of operations to process user input
chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Inputs: context from retriever, question directly from user
    | prompt  # Pass through the prompt template
    | llm  # Process with the ChatOllama model
    | StrOutputParser()  # Parse the output into a string format
)

# Interactive loop to continuously process user input until "quit" is entered
while True:
    inputt = input("Question: ")
    if inputt == "quit":
        break
    print(chain.invoke(inputt))  # Invoke the chain with user input and print the result
