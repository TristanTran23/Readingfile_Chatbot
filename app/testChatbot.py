import os
import sys
import langchain

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


loader = PyPDFLoader("./assets/sql-for-data-analysis-advanced-techniques-for-transforming-data-into-insights.pdf")
pages = loader.load()

chap1 = pages[11:32]