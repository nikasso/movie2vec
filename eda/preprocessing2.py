# Movie2Vec
# Preprocessing 2
# Preprocessing pipeline for MovieLens Latest dataset

import pandas as pd
import numpy as np
import cPickle as pickle


def listify(x):
    bad_list = ['']
    empty_list = []
    if x != bad_list:
        return x
    else:
        return empty_list

def preprocess():
    '''
    Preprocessing pipeline
    '''
    movies_meta = pd.merge(links, movies, on='movieId') # Merge movies w/links
    movies_meta = movies_meta.drop(['tmdbId'],axis=1) # Drop tmdb id column
    new_tags = tags.drop(['userId','timestamp'], axis=1)
    new_tags.tag = new_tags.tag.astype('str')
    grouped_tags = new_tags.groupby('movieId')['tag'].apply(lambda x: \
        '|'.join(x)).reset_index()
    movies_meta2 = pd.merge(movies_meta, grouped_tags, how='left', on='movieId')
    movies_meta2.tag = movies_meta2.tag.fillna(value='')
    movies_meta2.tag = movies_meta2.tag.apply(lambda x: x.lower())
    movies_meta2.genres = movies_meta2.genres.apply(lambda x: x.replace(' ','_'))
    movies_meta2.tag = movies_meta2.tag.apply(lambda x: x.replace(' ','_'))
    movies_meta2.genres = movies_meta2.genres.apply(lambda x: x.split('|'))
    movies_meta2.tag = movies_meta2.tag.apply(lambda x: x.split('|'))
    movies_meta2.tag = movies_meta2.tag.apply(lambda x: listify(x))
    movies_meta2['year'] = movies_meta2.title.apply(lambda x: x[-5:-1])
    movies_meta2.title = movies_meta2.title.apply(lambda x: x[:-7])
    movies_meta2['tags'] = movies_meta2.genres + movies_meta2.tag
    movies_meta2 = movies_meta2.drop(movies_meta2[['genres','tag']], axis=1)
    movies_meta2.tags = movies_meta2.tags.apply(lambda x: set(x))
    movies_meta2.tags = movies_meta2.tags.apply(lambda x: list(x))
    movies_meta2['tags_length'] = movies_meta2.tags.apply(lambda x: len(x))
    return movies_meta2

def get_data():
    '''
    Loads dataframes from MovieLens Latest csv files
    '''
    movies = pd.read_csv('../data/ml-latest/movies.csv')
    links = pd.read_csv('../data/ml-latest/links.csv')
    tags = pd.read_csv('../data/ml-latest/tags.csv')
    genome_tags = pd.read_csv('../data/ml-latest/genome-tags.csv')
    genome_scores = pd.read_csv('../data/ml-latest/genome-scores.csv')
    return movies, links, tags, genome_tags, genome_scores

if __name__ == '__main__':
    movies, links, tags, genome_tags, genome_scores = get_data()
    movie_df = preprocess()
    with open('../data/ml-latest_df.pkl', 'w') as f:
        pickle.dump(movie_df, f)
