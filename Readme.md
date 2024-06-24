# Book Recommender using Llama3 and RAG

## Description

The Streamlit based app implements a book recommendation functionality using an Kaggle data of Amazon's Book Collection (Custom Tuned) and implementation of RAG through pdf using Llama3 as generator, google genai embeddings as retriever and Groq API for inference acceleration.

## Notes

- The model is able to extract data from the provided data and understand it for user's QnA
- The model can suggest 100, top 10 and the best book from a particular category on user's question
- The model also occasionally provides the book's ratings and prices when needed.

## Things which can be improved

- The encoding could be better as Hindi and Malyalam text is encoded in garbage value format.
- The model is not able to conclude with a message after the user has asked the best book
- The application still relies on specific genres to be present in the query , which are present in the dataset such as genre "Science Fiction" would not get very accurate results which "Science, Fiction & Horror" will get.

## Video Demonstration

Watch the video demonstration of the project:

## Preview Video is stored here :

#Interface
![Image](https://github.com/Wriath18/PokeDex_Streamlit_LLM/blob/main/Images/Screenshot%202024-06-17%20135642.png)

## Categories Supported in accordnce with database

- Arts, Film & Photography
- Biographies, Diaries & True Accounts
- Business & Economics
- Children's Books
- Comics & Mangas
- Computing, Internet & Digital Media
- Crafts, Home & Lifestyle
- Crime, Thriller & Mystery
- Engineering
- Exam Preparation
- Fantasy, Horror & Science Fiction
- Health, Family & Personal Development
- Higher Education Textbooks
- History
- Language, Linguistics & Writing
- Law
- Literature & Fiction
- Medicine & Health Sciences
- Politics
- Reference
- Religion
- Romance
- School Books
- Science & Mathematics
- Sciences, Technology & Medicine
- Society & Social Sciences
- Sports
- Teen & Young Adult
- Textbooks & Study Guides
- Travel
