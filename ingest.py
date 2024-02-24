import os
import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import lance
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

def get_pdf_file_names(directory_name):
    path = directory_name + "\*.pdf"
    files = glob.glob(path)
    return files

def get_pdf_docs(pdf_file_names):
    all_pages = []
    for pdf_file_name in pdf_file_names:
        loader = PyPDFLoader(pdf_file_name)
        pages = loader.load_and_split()
        all_pages.append(pages)
    return pages #TODO

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len)
    chunks = text_splitter.split_documents(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def ingest(docs):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_pdf_docs2(directory_name) :
    loader = PyPDFDirectoryLoader(directory_name)
    docs = loader.load_and_split()
    return docs

# get pdf file names
pdf_file_names = get_pdf_file_names("C:\\Users\\nick.luu\\OneDrive - Volaris Group\\Documents\\Chatbot")

# get pdf docs
#docs = get_pdf_docs(pdf_file_names)
docs = get_pdf_docs2("C:\\Users\\nick.luu\\OneDrive - Volaris Group\\Documents\\Chatbot")

# get text chunks
#chunks = get_text_chunks(raw_text)
#st.write(text_chunks)

# create vector store
#vectorstore = get_vectorstore(text_chunks)

db = ingest(docs)
db.save_local(".aeros_docs")


#print(get_files("C:\\Users\\nick.luu\\OneDrive - Volaris Group\\Documents\\Chatbot"))