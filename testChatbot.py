import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
def readFile(question):
    local_path = "./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf"
    loader = PyPDFLoader(file_path=local_path)
    data = loader.load()
    chap1 = data[11:32]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=0,
        separators=["\n", "\n\n"]
    )
    text_chunks = text_splitter.split_documents(chap1)

    # embeddings=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    vector_store = Chroma.from_documents(
        documents=text_chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag",
        persist_directory="./chroma_db"
    )
    vector_store.persist()

    # embedding_file = open("embeding.txt", "w")
    # for list1 in embeddings:
    #     for i in list1:
    #         print(i, file = embedding_file)
            
    # embedding_file.close()

    local_model = "llama3"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["query"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_store.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT,
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    return(chain.invoke(question))
    
demo = gr.Interface(fn=readFile, inputs="text", outputs="text")
demo.launch()