import gradio as gr
import langchain
import sys
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

loader = PyPDFLoader("./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf")
pages = loader.load()
chap1 = pages[11:32]

#Helper function
pipe = pipeline("text-generation", model="microsoft/Phi-3-medium-128k-instruct", trust_remote_code=True)

#Load model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-medium-128k-instruct", trust_remote_code=True)

