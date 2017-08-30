# Movie2Vec
# Movie2Vec Class
# Main model script, containing class with all functions for building & saving
# models, loading models, get recommendations with models

import pandas as pd
import numpy as np
import cPickle as pickle
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Movie2Vec(object):

    def __init__(self):
        with open('../data/ml-latest_df.pkl') as f:
            self.movie_df = pickle.load(f) # Movie dataframe
        self.sentences = self.movie_df.tags.tolist() # List of 'sentences',
            # sentence for each movie
        self.max_sentence_size = self.movie_df.tags_length.max()

    def build_w2v_model(self, filename='w2v_model', **w2v_params):
        '''
        Builds Word2Vec model trained to sentences, saves model.
        Also gets tag vectors from model.

        Input:
        ------
        filename: str of filename you want to give built model when saving
            (default is 'w2v_model')
        w2v_params: custom parameters for Word2Vec model
        '''
        print '\nTraining Word2Vec model...'
        # Fit model to sentences
        self.w2v_model = Word2Vec(self.sentences, **w2v_params)
        print 'Word2Vec model trained.'
        self.w2v_model.save(filename)
        self.tag_vectors = self.w2v_model.wv

    def build_d2v_model(self, filename='d2v_model', **d2v_params):
        '''
        Builds Doc2Vec model trained to sentences, saves model.
        Also gets movie vectors from model.

        Input:
        ------
        filename: str of filename you want to give built model when saving
            (default is 'd2v_model')
        d2v_params: custom parameters for Doc2Vec model
        '''
        tagged_sentences = []
        for i, sentence in enumerate(self.sentences):
            u_sentence = [unicode(x, 'utf-8') for x in sentence]
            tagged_sentences.append(TaggedDocument(u_sentence,[self.movie_df.title[i]]))
        print '\nTraining Doc2Vec model...'
        # Fit model to sentences
        self.d2v_model = Doc2Vec(tagged_sentences)
        print 'Doc2Vec model trained.'
        self.d2v_model.save(filename)
        self.movie_vectors = self.d2v_model.docvecs

    def load_w2v_model(self, filename):
        '''
        Loads pre-trained Word2Vec model saved under filename.
        Also gets tag vectors from model.

        Input:
        ------
        filename: str of filename
        '''
        self.w2v_model = Word2Vec.load(filename)
        self.tag_vectors = self.w2v_model.wv

    def load_d2v_model(self, filename):
        '''
        Loads pre-trained Doc2Vec model saved under filename.
        Also gets movie vectors from model.

        Input:
        ------
        filename: str of filename
        '''
        self.d2v_model = Doc2Vec.load(filename)
        self.movie_vectors = self.d2v_model.docvecs

    def recommend_tags(self, pos_tags=[], neg_tags=[], num_recs=10):
        '''
        Uses Word2Vec most_similar method to return list of most similar tags.

        Input:
        ------
        pos_tags: list of strings of positive tags (tags to add)
        neg_tags: list of strings of negative tags (tags to subtract)
        num_recs: int of num recommendations desired (default is 10)

        Output:
        -------
        similar_tags: list of tuples of similar tags and their probabilities
        '''
        similar_tags = self.tag_vectors.most_similar(positive=pos_tags, \
            negative=neg_tags, topn=num_recs)
        # Convert from ASCII to Unicode
        u_similar_tags = []     # new list of tuples where tag strings
                                # will be in unicode
        for tag, probab in similar_tags:
            u_tag = tag.decode('utf-8')
            u_similar_tags.append((u_tag, probab))
        return u_similar_tags

    def recommend_movies(self, pos_movies=[], neg_movies=[], num_recs=10):
        '''
        Uses Doc2Vec most_similar method to return list of most similar movies.

        Input:
        ------
        pos_movies: list of indices of positive tags (tags to add)
        neg_tags: list of indices of negative tags (tags to subtract)
        num_recs: int of num recommendations desired (default is 10)

        Output:
        -------
        similar_movies: list of tuples of similar movies and their probabilities
        '''
        similar_movies = self.movie_vectors.most_similar(positive=pos_movies, \
            negative=neg_movies, topn=num_recs)
        # Convert from ASCII to Unicode
        u_similar_movies = []   # new list of tuples where movie title strings
                                # will be in unicode
        for movie, probab in similar_movies:
            u_movie = movie.decode('utf-8')
            u_similar_movies.append((u_movie, probab))
        return u_similar_movies

    def parse_input(self, in_string):
        '''
        Input: string(s) of movie(s) entered by user - str: in_string
        (could replace in_string with pos_string and neg_string later)

        Output: list(s) of indices of movies in movie_df that can be given to model
        - lst: pos_movies and neg_movies
        '''
        pos_movies = [] # list of indices of movies to add
        neg_movies = [] # list of indices of movies to subtract
        movies_lst = in_string.split('|')
        for movie in movies_lst:
            movie_idx = self.movie_df.loc[self.movie_df.title == movie].index[0]
            pos_movies.append(movie_idx)
        return pos_movies


if __name__ == '__main__':
    # #main()
    # # Load dataframe
    # with open('../data/ml-latest_df.pkl') as f:
    #     movie_df = pickle.load(f)
    #
    # in_string = "Godfather, The|Avatar" # Example of movies entered by user
    # pos_movies = parse_input(in_string) # [842, 14632] for example
    #
    # #m2v = Movie2Vec()
    pass
