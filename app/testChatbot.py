import gradio as gr
import openai
import langchain
import sys
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

loader = PyPDFLoader("./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf")
pages = loader.load()
chap1 = pages[11:32]

