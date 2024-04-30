import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

pdf_folder_path = os.path.abspath('./whitePaperDoc')

# Check if the directory exists
if not os.path.isdir(pdf_folder_path):
    print(f"Error: Directory {pdf_folder_path} not found")
    exit()

# read data from the file and put them into a variable called raw_text
dir_path = './NewWhitePaper/'

# read data from all PDF files in the directory
raw_text = ''
for file_name in os.listdir(dir_path):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create an FAISS index for information retrieval
docsearch = FAISS.from_texts(texts, embeddings)
# Save to local
docsearch.save_local("NewDB")

