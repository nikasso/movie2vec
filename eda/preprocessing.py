# Movie2Vec
# Preprocessing
# Modifying df to make it palatable to word2vec model

import pandas as pd
from eda import imdb_eda

def preprocess(movies):
    '''
    Input: IMDB df with columns movie_title, director_name, genres, and
    plot_keywords

    Creates 'sentences' for each movie, with words being all the tags from
    columns of df
    '''
    movies.director_name = movies.director_name.apply(lambda x: \
        x.replace(' ','_'))
    movies.director_name = movies.director_name.apply(lambda x: [x])
    movies['tags'] = movies.director_name + movies.genres + \
        movies.plot_keywords
    movies.drop(movies.columns[[1,2,3]], axis=1, inplace=True)
    return movies

if __name__ == '__main__':
    imdb_df = imdb_eda()
    movies = preprocess(imdb_df)
