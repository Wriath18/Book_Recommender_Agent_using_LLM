# import pandas as pd

# # Load your dataset
# df = pd.read_csv('Books_df.csv')
# df = df.drop(columns=['No. of People rated', 'URLs', 'Type', 'UnTitled: 0'], axis=1)
# genre_list = df['Main Genre'].unique()
# df.to_csv('processed_books_dataset.csv', index=False)

# books_list = df.to_dict('records')
# books_json = df.to_json(orient='records', lines=True)

