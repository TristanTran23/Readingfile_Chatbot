# Import necessary modules and classes from LangChain library
import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA, SequentialChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

vector_store = Chroma(
    persist_directory=".//chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
)

# Initialize a ChatOllama language model
local_model = "llama3"
llm = ChatOllama(model=local_model, temperature = 0)

# Define a prompt template for generating alternative versions of user queries
QUERY_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""you are an AI teacher assistant. you must not make any assumptions. you must use the information given by my pdf files. 
        you are required to find at least 5 paragraphs related to the user's question in this textbook. each paragraph must directly answer the question or give helpful information.
        provide the answer in seperated paragraphs. Provide these alternative questions separated by newlines.
        Original question: {question}""",
)

#Compress the document using the LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(),
    prompt=QUERY_PROMPT,
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
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following question to english:"
    "\n\n{question}"
)

second_prompt = ChatPromptTemplate.from_template(
    """you are an AI teacher assistant. you must not make any assumptions. you must use the information given by my pdf files.
    you are required to find at least 5 paragraphs related to the user's question in this textbook. each paragraph must directly answer the question or give helpful information.
    provide the answer in seperated paragraphs. Provide these alternative questions separated by newlines.
    
    Original question: {english_question}"""
)

prompt = ChatPromptTemplate.from_template(template)

# Define a chain of operations to process user input
# chain_1 = LLMChain(
#     llm=llm, prompt=first_prompt, 
#     output_key="english_question"
# )
chain_2 = RetrievalQA(retriever=retriever, prompt=second_prompt)

chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}  # Inputs: context from retriever, question directly from user
    | prompt  # Pass through the prompt template
    | llm  # Process with the ChatOllama model
    | StrOutputParser()  # Parse the output into a string format
)

# overall_chain = SequentialChain(
#     chains=[chain_1, chain],
#     input_keys=["question"],
#     output_keys=["answer"],
# )


def give_ans(question):
    return chain.invoke(question)

demo = gr.Interface(fn=give_ans, inputs="text", outputs="text")
demo.launch()
