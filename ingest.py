import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

def get_pdf_file_names(directory_name):
    path = directory_name + "\*.pdf"
    files = glob.glob(path)
    return files

def ingest(docs):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_pdf_docs(directory_name) :
    loader = PyPDFDirectoryLoader(directory_name)
    docs = loader.load_and_split()
    return docs

# environemnt variables
load_dotenv()
pdf_docs_dictory = os.environ['PDF_DOCS_DIRECTORY']
vector_database_name = os.environ['VECTOR_DATABASE_NAME']

# get pdf file names
pdf_file_names = get_pdf_file_names(pdf_docs_dictory)

# get pdf docs
docs = get_pdf_docs(pdf_docs_dictory)

# ingest the docs
db = ingest(docs)

# persist to database
db.save_local(vector_database_name)
