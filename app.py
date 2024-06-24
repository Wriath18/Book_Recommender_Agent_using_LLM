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



import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

# Step 1: Prepare and Process Dataset

# Load your dataset
df = pd.read_csv('Books_df.csv')

# Extract relevant columns
books_df = df[['Title', 'Author', 'Main Genre', 'Rating']]

# Step 2: Generate Embeddings and Setup FAISS Index

# Load a sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for book descriptions
book_descriptions = (books_df['Title'] + " " + books_df['Author']).tolist()
embeddings = model.encode(book_descriptions)

# Check embeddings dimension
print("Embeddings dimension:", embeddings.shape)

# Initialize FAISS index with the correct dimension
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, 'books_index.faiss')

# Step 3: Integrate with Streamlit

# Load the dataset as a list of dictionaries
books_list = books_df.to_dict('records')

st.title("Book Recommendation Agent with RAG")

genre_list = books_df['Main Genre'].unique()
query = None
for i in genre_list:
    if st.button(i):
        query = i
        break



# Step 1: User Input
if query:
    num_books = st.slider("Number of top books to fetch:", 10, 100, 100)

    if st.button("Fetch Top Books"):
        # Step 2: Retrieve relevant books using FAISS
        query_embedding = model.encode([query])
        _, indices = index.search(query_embedding, num_books)
        retrieved_books = [books_list[i] for i in indices[0]]

        st.write(f"Top {num_books} books based on your query:")
        for book in retrieved_books:
            st.write(f"- {book['Title']} by {book['Author']} (Rating: {book['Rating']})")

        # Step 3: Narrow Down to Top 10
        top_10_books = sorted(retrieved_books, key=lambda x: x['Rating'], reverse=True)[:10]
        st.write(f"\nTop 10 books based on Rating:")
        for i, book in enumerate(top_10_books):
            st.write(f"{i+1}. {book['Title']} by {book['Author']} (Rating: {book['Rating']})")

        # Step 4: User Selects One Book from Top 10
        selected_book_index = st.selectbox("Select a book from the top 10:", range(1, 11)) - 1
        selected_book = top_10_books[selected_book_index]
        st.write(f"\nYou selected: {selected_book['Title']} by {selected_book['Author']} (Rating: {selected_book['Rating']})")

        # Step 5: Conclude the Workflow
        st.success("Thank you for using the book recommendation agent!")
