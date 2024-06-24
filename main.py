import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_BOOKS_API = os.getenv('GOOGLE_BOOKS_API')
BASE_URL = os.getenv('BASE_URL')
MAX_RESULTS = 40



def get_books(genre, num_books):
    books = []
    for start in range(0, num_books, MAX_RESULTS):
        params = {
            'q':f'subject{genre}',
            'startindex':start,
            'maxResults' : MAX_RESULTS,
            'key' : GOOGLE_BOOKS_API

        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            items = response.json().get('items', [])
            books.extend(items)

        else:
            st.error("Error fetching results from google")
            break
    return books[:num_books]

st.title("Gertie")

genre = st.text_input("Eenter a genre (eg. fiction)", "fiction")
num_books = st.slider("Number of book to fetch", 10, 50,100)


if st.button("fetch top books"):
    books = get_books(genre, num_books=num_books)
    if books:
        st.write(f"Top {num_books} in the given {genre} genre")
        for book in books:
            st.write(f" - {book['volumeInfo'].get('title', 'Unknown Title')} by {','.join(book['volumeInfo'].get('authors', ['Unknown Author']))}")

        top_10_books = books[:10]
        st.write(f"\nTop 10 boks in {genre} genre")
        for i, book in enumerate(top_10_books):
            st.write(f"{i+1}. {book['volumeInfo'].get('title', 'Unknown Title')} by {','.join(book['volumeInfo'].get('authors', ['Unknown Author']))}")

        selected_book_index = st.selectbox('Select book from the top 10', range(1,11)) - 1
        selectedbook = top_10_books[selected_book_index]
        st.write(f"You selectee : {selectedbook['volumeInfo'].get('title', 'Unknown Title')} by {','.join(selectedbook['volumeInfo'].get('authors', ['Unknown Author']))}")


    st.success("Thank you now go")

else:
    st.warning("none book found")
