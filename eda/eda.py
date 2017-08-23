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
    movies_meta2 = pd.merge(movies_meta, avg_ratings, how='left', on='movieId')
    # Modify tags df to group tags by movie
    new_tags = tags.drop(['userId','timestamp'], axis=1)
    grouped_tags = new_tags.groupby('movieId').agg(lambda x: ' '.join(x)).reset_index()
    # Adding columns above to greater movie df
    movies_meta3 = pd.merge(movies_meta2, grouped_tags, how='left', on='movieId')
    return movies_meta3

def imdb_eda():
    '''
    EDA for IMDB 5000 dataset, cleaning and manipulating columns into workable
    table.
    '''
    # Load movies df
    movies = pd.read_csv('../data/imdb5000.csv')
    # Make genre column values lists of genres
    movies2 = movies.copy()
    movies2.genres = movies2.genres.apply(lambda x: x.split('|'))
    # Replace null values in plot_keywords with empty strings, then made them lists
    movies2.plot_keywords = movies2.plot_keywords.fillna(value='')
    movies2.plot_keywords = movies2.plot_keywords.apply(lambda x: x.split('|'))
    ##return movies2
    # Create a subset of movies df with only a few columns (mostly text) so
    # we can bin/dummify/one-hot-encode
    movies3 = movies2.copy()
    movies3 = movies3[['movie_title','director_name','actor_1_name', \
        'actor_2_name','actor_3_name','genres','plot_keywords', 'country', \
        'content_rating','title_year']]
    # Only retain movies with directors & years listed - movies without directors seem
    # to be TV shows, not movies, only 4 rows w/o year
    movies3 = movies3[movies3['director_name'].notnull()]
    movies3 = movies3[movies3['title_year'].notnull()]
    ##return movies3
    # Creating an even smaller df with a subset of columns that are all text,
    # with no null values (4,935 rows) to do begin testing out word2vec
    movies4 = movies3.copy()
    movies4 = movies3[['movie_title','director_name','genres','plot_keywords']]
    return movies4

if __name__ == '__main__':
    ml_latest_small_df = ml_eda()
    imdb5000_df = imdb_eda()
