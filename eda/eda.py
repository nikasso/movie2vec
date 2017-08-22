# Movie2Vec
# EDA
# Functions for initial data exploration, cleaning, and manipulating

import pandas as pd

def ml_eda():
    '''
    EDA for MovieLens Latest small dataset, cleaning and combining 4 tables
    into one workable dataframe.
    '''
    # Load dataframes from MovieLens csv files
    links = pd.read_csv('../data/ml-latest-small/links.csv')
    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')
    tags = pd.read_csv('../data/ml-latest-small/tags.csv')

    # Joining links & movies on movieId
    movies_meta = pd.merge(links, movies, how='left', on='movieId')

    # Creating df of movies and their avg ratings
    avg_ratings = ratings.groupby(['movieId'], as_index=False)['rating'].mean()
    avg_ratings = avg_ratings.rename(columns={'rating': 'avg_rating'})

    # Adding columns above to greater movie df
    movies_meta2 = pd.merge(movies_meta, avg_ratings, how='left' on='movieId')

    # Modify tags df to group tags by movie
    new_tags = tags.drop(['userId','timestamp'], axis=1)
    grouped_tags = new_tags.groupby('movieId').agg(lambda x: ' '.join(x)).reset_index()

    # Adding columns above to greater movie df
    movies_meta3 = pd.merge(movies_meta2, grouped_tags, how='left', on='movieId')
