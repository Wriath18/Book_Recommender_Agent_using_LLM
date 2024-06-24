# # # import streamlit as st
# # # import requests
# # # from dotenv import load_dotenv
# # # import os

# # # load_dotenv()

# # # GOOGLE_BOOKS_API = os.getenv('GOOGLE_BOOKS_API')
# # # BASE_URL = os.getenv('BASE_URL')
# # # MAX_RESULTS = 40



# # # def get_books(genre, num_books):
# # #     books = []
# # #     for start in range(0, num_books, MAX_RESULTS):
# # #         params = {
# # #             'q':f'subject{genre}',
# # #             'startindex':start,
# # #             'maxResults' : MAX_RESULTS,
# # #             'key' : GOOGLE_BOOKS_API

# # #         }

# # #         response = requests.get(BASE_URL, params=params)

# # #         if response.status_code == 200:
# # #             items = response.json().get('items', [])
# # #             books.extend(items)

# # #         else:
# # #             st.error("Error fetching results from google")
# # #             break
# # #     return books[:num_books]

# # # st.title("Gertie")

# # # genre = st.text_input("Eenter a genre (eg. fiction)", "fiction")
# # # num_books = st.slider("Number of book to fetch", 10, 50,100)


# # # if st.button("fetch top books"):
# # #     books = get_books(genre, num_books=num_books)
# # #     if books:
# # #         st.write(f"Top {num_books} in the given {genre} genre")
# # #         for book in books:
# # #             st.write(f" - {book['volumeInfo'].get('title', 'Unknown Title')} by {','.join(book['volumeInfo'].get('authors', ['Unknown Author']))}")

# # #         top_10_books = books[:10]
# # #         st.write(f"\nTop 10 boks in {genre} genre")
# # #         for i, book in enumerate(top_10_books):
# # #             st.write(f"{i+1}. {book['volumeInfo'].get('title', 'Unknown Title')} by {','.join(book['volumeInfo'].get('authors', ['Unknown Author']))}")

# # #         selected_book_index = st.selectbox('Select book from the top 10', range(1,11)) - 1
# # #         selectedbook = top_10_books[selected_book_index]
# # #         st.write(f"You selectee : {selectedbook['volumeInfo'].get('title', 'Unknown Title')} by {','.join(selectedbook['volumeInfo'].get('authors', ['Unknown Author']))}")


# # #     st.success("Thank you now go")

# # # else:
# # #     st.warning("none book found")
# # import pandas as pd
# # import pdfkit

# # # Read the CSV file into a pandas DataFrame
# # df = pd.read_csv('D:\Programming-1\Book_Recommender_Agent_using_LLM\processed_books_dataset.csv')

# # # Convert the DataFrame to an HTML string
# # html_table = df.to_html(index=False)

# # # Convert the HTML to PDF using pdfkit
# # pdfkit.from_string(html_table, 'output.pdf')
# import streamlit as st
# import os
# import csv
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import CSVLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = CSVLoader("./Books_df.csv")
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# def main():
#     st.title("CSV Answering Bot")

#     if "embedding_done" not in st.session_state:
#         st.session_state.embedding_done = False

#     if "answer_generated" not in st.session_state:
#         st.session_state.answer_generated = False

#     prompt1 = st.text_input("Enter Your Question")

#     if prompt1 and not st.session_state.embedding_done:
#         vector_embedding()
#         st.write("Vector Store DB is ready")
#         st.session_state.embedding_done = True

#     if prompt1 and st.session_state.embedding_done and not st.session_state.answer_generated:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         st.session_state.answer = response['answer']
#         st.session_state.answer_generated = True

#     if st.session_state.embedding_done and st.session_state.answer_generated:
#         st.write("Answer:")
#         st.write(st.session_state.answer)

#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")

# if __name__ == '__main__':
#     load_dotenv()
#     groq_api_key = os.getenv('GROQ_API_KEY')
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#     llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#     prompt = ChatPromptTemplate.from_template(
#         """
#         Answer the questions based on the provided context only.
#         Please provide the most accurate response based on the question.
#         <context>
#         {context}
#         <context>
#         Questions: {input}
#         """
#     )

#     main()


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

from transformers import pipeline
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
    st.text("Upload your image")

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
    # response = retrieval_chain.invoke({'input': f"Describe {result}"})
    # st.session_state.description = response['answer']
    # st.session_state.description_generated = True

    if st.session_state.embedding_done:
        st.title("Llama3 Powered Pokedex RAG Q&A")

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

            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

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

