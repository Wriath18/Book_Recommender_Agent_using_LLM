
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import fitz

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./") 
        st.session_state.docs = st.session_state.loader.load() 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500) 
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  

def main():
    st.title("Book Recommendation System using Llama3 (with Groq) and RAG")
    if "embedding_done" not in st.session_state:
        st.session_state.embedding_done = False

    if "description_generated" not in st.session_state:
        st.session_state.description_generated = False


            
    vector_embedding()
    st.write("Vector Store DB is ready")
    st.session_state.embedding_done = True

    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    

    if st.session_state.embedding_done:
        st.title("Llama3 Powered Search")

        if st.session_state.description_generated:
            st.write("Description of the classified Pokémon:")
            st.write(st.session_state.description)

        prompt1 = st.text_input("Enter Your Question From Documents")

        if prompt1:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])

            

if __name__ == '__main__':
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    main()

