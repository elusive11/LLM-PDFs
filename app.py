# pip install langchain streamlit load_dotenv PyPDF2 langchain_openai faiss-cpu pypdf
# https://python.langchain.com/docs/integrations/vectorstores/faiss

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain_community.vectorstores import FAISS
from langchain_community.llms import predictionguard


def get_retriver():
    vector_database_name = os.environ['VECTOR_DATABASE_NAME']
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(vector_database_name, embeddings)
    return db.as_retriever()
    

def get_conversation_chain(retriever):
    llm = predictionguard
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return conversation_chain    
    

def handle_userinput(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    # environemnt variables
    load_dotenv()

    st.set_page_config(
        page_title="Chat",
        page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if (st.button("Process")):
            with st.spinner("Processing"):
                # load retriever
                st.session_state.retriever = get_retriver()

                # create conversation
                st.session_state.conversation = get_conversation_chain(st.session_state.retriever)


if __name__ == '__main__':
    main()
