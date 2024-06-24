# # import faiss
# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # from sentence_transformers import SentenceTransformer
# # from preprocessing import genre_list, books_json, books_list, df

# # model = SentenceTransformer('all-MiniLM-L6-v2')

# # book_des = df['Title'] + " " + df['Author']
# # embeddings = model.encode(book_des.tolist())


# # dimension = embeddings.shape[0]
# # index = faiss.IndexFlatL2(dimension)
# # index.add(embeddings)

# # faiss.write_index(index, 'books_index.faiss')

# # print("index loaded")
# # index = faiss.read_index('books_index.faiss')
# # st.title("Book Recommendation Agent")

# # query = st.text_input("Enter a book description or genre ()")
# # num_books = st.slider("NUmber of top books : ", 10, 100, 100)

# # if st.button("Fetch top books"):
# #     query_embedding = model.selection([query])
# #     _, indices = index.search(query_embedding, num_books)

# #     retireved = [books_list[i] for i in indices[0]]

# #     top_10_books = sorted(retireved, key=lambda x:x['Rating'], reverse=True)[:10]
# #     st.write(f"Top {num_books} for your query : ")
# #     for book in retireved:
# #         st.write(f"- {book['Title']} by {book['Author']} (Rating: {book['Rating']}, Genre: {book['Main Genre']})")

# #     selected_books_index = st.selection("Select the nbook index", range(1,11)) -1 
# #     selected_book = top_10_books[selected_books_index]
# #     st.write(f"\nYou selected: {selected_book['Title']} by {selected_book['Author']} (Rating: {selected_book['Yating']}, Genre: {selected_book['Main Genre']})")


# #     st.success("Done ")



# import pandas as pd
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import streamlit as st
# import os
# import pickle


# df = pd.read_csv('Books_df.csv')

# books_df = df[['Title', 'Author', 'Main Genre', 'Rating']]

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings_path = 'book_embeddings.pkl'
# index_path = 'books_index.faiss'


# if os.path.exists(embeddings_path) and os.path.exists(index_path):
#     # Load the embeddings and FAISS index if they exist
#     with open(embeddings_path, 'rb') as f:
#         embeddings = pickle.load(f)
#     index = faiss.read_index(index_path)
# else:
#     # Generate embeddings for book descriptions
#     book_descriptions = (books_df['Title'] + " " + books_df['Author']).tolist()
#     embeddings = model.encode(book_descriptions)

#     # Save embeddings to disk
#     with open(embeddings_path, 'wb') as f:
#         pickle.dump(embeddings, f)

#     # Initialize FAISS index with the correct dimension
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)

#     # Save the index to disk
#     faiss.write_index(index, index_path)

# books_list = books_df.to_dict('records')

# st.title("Book Recommendation Agent with RAG")

# genre_list = books_df['Main Genre'].unique()
# query = None
# # for i in genre_list:
# #     if st.button(i):
# #         query = i
# #         break

# query = st.selectbox("Select a book from the top 10:",genre_list)
# retrieved_books = None


# # Step 1: User Input
# if query:
#     # num_books = st.slider("Number of top books to fetch:", 10, 100, 100)
#     num_books = 100
#     if st.button("Fetch Top Books"):
#         query_embedding = model.encode([query])
#         _, indices = index.search(query_embedding, num_books)
#         retrieved_books = [books_list[i] for i in indices[0]]

#         st.write(f"Top {num_books} books based on your query:")
#         for book in retrieved_books:
#             st.write(f"- {book['Title']} by {book['Author']} (Rating: {book['Rating']})")

        
#         top_10_books = sorted(retrieved_books, key=lambda x: x['Rating'], reverse=True)[:10]
#         st.write(f"\nTop 10 books based on Rating:")
#         for i, book in enumerate(top_10_books):
#             st.write(f"{i+1}. {book['Title']} by {book['Author']} (Rating: {book['Rating']})")


#         selected_book_index = st.selectbox("Select a book from the top 10:", range(1, 11)) - 1
#         selected_book = top_10_books[selected_book_index]
#         st.write(f"\nYou selected: {selected_book['Title']} by {selected_book['Author']} (Rating: {selected_book['Rating']})")

 
#         st.success("Thank you for using the book recommendation agent!")




# import os
# import pandas as pd
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import pickle
# from dotenv import load_dotenv
# import time


# load_dotenv()
# groq_api = os.getenv('GROQ_API_KEY')


# def main():
#     st.title("LLM Book Recommender RAG")

#     df = pd.read_csv('Books_df.csv')
#     books_df = df[['Title', 'Author', 'Main Genre', 'Rating']]

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings_path = 'book_embeddings.pkl'
#     index_path = 'books_index.faiss'


#     if os.path.exists(embeddings_path) and os.path.exists(index_path):
#         # Load the embeddings and FAISS index if they exist
#         with open(embeddings_path, 'rb') as f:
#             embeddings = pickle.load(f)
#         index = faiss.read_index(index_path)
#     else:
#         # Generate embeddings for book descriptions
#         book_descriptions = (books_df['Title'] + " " + books_df['Author']).tolist()
#         embeddings = model.encode(book_descriptions)

#         # Save embeddings to disk
#         with open(embeddings_path, 'wb') as f:
#             pickle.dump(embeddings, f)

#         # Initialize FAISS index with the correct dimension
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(embeddings)

#         # Save the index to disk
#         faiss.write_index(index, index_path)

#     books_list = books_df.to_dict('records')
#     genre_list = books_df['Main Genre'].unique()
#     query = None
#     # for i in genre_list:
#     #     if st.button(i):
#     #         query = i
#     #         break

#     query = st.selectbox("Select a book from the top 10:",genre_list)
#     retrieved_books = None


#     # Step 1: User Input
#     if query:
#         # num_books = st.slider("Number of top books to fetch:", 10, 100, 100)
#         num_books = 100
#         if st.button("Fetch Top Books"):
#             query_embedding = model.encode([query])
#             _, indices = index.search(query_embedding, num_books)
#             retrieved_books = [books_list[i] for i in indices[0]]

#             st.write(f"Top {num_books} books based on your query:")
#             for book in retrieved_books:
#                 st.write(f"- {book['Title']} by {book['Author']} (Rating: {book['Rating']})")

#             context = "\n".join([f"{i+1}, {book['Title']} by {book['Author']} (Rating : {book['Rating']})" for i,book in enumerate(retrieved_books)])
#             print("Context got it")
#             st.session_state['context'] = context

#         if 'context' in st.session_state:
#             st.text_input("Enter your query : ", key='prompt1')

#             if st.session_state.prompt1:  
#                 if "llm" not in st.session_state:
#                     print("LLm got it")
#                     st.session_state.llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

#                 prompt_template = ChatPromptTemplate.from_template(
#                     """
#                     Based on the following list of books, find the top 10 books with the highest ratings and answer any user questions about them.
#                     if you don't know the answer, response i don't know about that
#                     <context>
#                     {context}
#                     <context>
#                     Questions: {input}
#                     """
#                 )

#                 prompt = prompt_template.format(context=st.session_state['context'], input=st.session_state.prompt1)
#                 document_chain = create_stuff_documents_chain(st.session_state.llm, prompt_template)
#                 start = time.process_time()
#                 response = document_chain.invoke({'input': st.session_state.prompt1, 'context': st.session_state['context']})
#                 st.write(f"Response time: {time.process_time() - start} seconds")
#                 st.write(response['answer'])

#                 # with st.expander("Document Similarity Search"):
#                 #     for i, doc in enumerate(response['context']):
#                 #         st.write(doc.page_content)
#                 #         st.write("----------------------------------------")
#             else:
#                 print("Prompt did not take it")

# if __name__ == '__main__':
#     load_dotenv()
#     groq_api = os.getenv('GROQ_API_KEY')
#     main()

import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
import pickle
from dotenv import load_dotenv
import time


load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')


def main():
    st.title("LLM Book Recommender RAG")

    df = pd.read_csv('Books_df.csv')
    books_df = df[['Title', 'Author', 'Main Genre', 'Rating']]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_path = 'book_embeddings.pkl'
    index_path = 'books_index.faiss'

    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(index_path)
    else:
        book_descriptions = (books_df['Title'] + " " + books_df['Author']).tolist()
        embeddings = model.encode(book_descriptions)

        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)

    books_list = books_df.to_dict('records')
    genre_list = books_df['Main Genre'].unique()
    query = st.selectbox("Select a genre:", genre_list)
    retrieved_books = None

    if query:
        num_books = 100
        if st.button("Fetch Top Books"):
            query_embedding = model.encode([query])
            _, indices = index.search(query_embedding, num_books)
            retrieved_books = [books_list[i] for i in indices[0]]

            st.write(f"Top {num_books} books based on your query:")
            for book in retrieved_books:
                st.write(f"- {book['Title']} by {book['Author']} (Rating: {book['Rating']})")

            documents = [Document(page_content=f"{book['Title']} by {book['Author']} (Rating: {book['Rating']})", metadata={"Title": book['Title'], "Author": book['Author'], "Rating": book['Rating']}) for book in retrieved_books]
            st.session_state['documents'] = documents
            print(type(documents))

        if 'documents' in st.session_state:
            st.text_input("Enter your query:", key='prompt1')

            if st.session_state.prompt1:  
                if "llm" not in st.session_state:
                    st.session_state.llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

                prompt_template = ChatPromptTemplate.from_template(
                    """
                    Based on the following list of books, find the top 10 books with the highest ratings and answer any user questions about them.
                    If you don't know the answer, respond with "I don't know about that."
                    <context>
                    {context}
                    <context>
                    Questions: {input}
                    """
                )

                context = "\n".join([doc.page_content for doc in st.session_state['documents']])
                prompt = prompt_template.format(context=context, input=st.session_state.prompt1)
                document_chain = create_stuff_documents_chain(st.session_state.llm, prompt_template)
                start = time.process_time()
                response = document_chain.invoke({'input': st.session_state.prompt1, 'context': context})
                st.write(f"Response time: {time.process_time() - start} seconds")
                st.write(response['answer'])

if __name__ == '__main__':
    load_dotenv()
    groq_api = os.getenv('GROQ_API_KEY')
    main()












