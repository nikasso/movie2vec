# Movie2Vec
# Movie2Vec Class
# Main model script, containing class with all functions for building & saving
# models, loading models, get recommendations with models

import pandas as pd
import numpy as np
import cPickle as pickle
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cosine


class Movie2Vec(object):

    def __init__(self):
        with open('../data/ml-latest_df.pkl') as f:
            self.movie_df = pickle.load(f) # Movie dataframe
        self.sentences = self.movie_df.tags.tolist() # List of 'sentences',
            # sentence for each movie
        self.max_sentence_size = self.movie_df.tags_length.max()

    def build_w2v_model(self, filepath='models/w2v_model', **w2v_params):
        '''
        Builds Word2Vec model trained to sentences, saves model.
        Also gets tag vectors from model.

        Input:
        ------
        filepath: str of filepath you want to give to built model when saving
            (default is 'models/w2v_model')
        w2v_params: custom parameters for Word2Vec model
        '''
        print '\nTraining Word2Vec model...'
        # Fit model to sentences
        self.w2v_model = Word2Vec(self.sentences, **w2v_params)
        print 'Word2Vec model trained.'
        #self.w2v_model.save('models/{}'.format(filename))
        self.w2v_model.save(filepath)
        self.tag_vectors = self.w2v_model.wv
        self.vectorize_movies()

    def build_d2v_model(self, filepath='models/d2v_model', **d2v_params):
        '''
        Builds Doc2Vec model trained to sentences, saves model.
        Also gets movie vectors from model.

        Input:
        ------
        filepath: str of filepath you want to give to built model when saving
            (default is 'models/d2v_model')
        d2v_params: custom parameters for Doc2Vec model
        '''
        tagged_sentences = []
        for i, sentence in enumerate(self.sentences):
            u_sentence = [unicode(x, 'utf-8') for x in sentence]
            tagged_sentences.append(TaggedDocument(u_sentence,[self.movie_df.title[i]]))
        print '\nTraining Doc2Vec model...'
        # Fit model to sentences
        self.d2v_model = Doc2Vec(tagged_sentences, **d2v_params)
        print 'Doc2Vec model trained.'
        self.d2v_model.save(filepath)
        self.movie_vectors = self.d2v_model.docvecs

    def load_w2v_model(self, filepath):
        '''
        Loads pre-trained Word2Vec model saved under filename.
        Also gets tag vectors from model.

        Input:
        ------
        filepath: str of file path
        '''
        self.w2v_model = Word2Vec.load(filepath)
        self.tag_vectors = self.w2v_model.wv
        self.vectorize_movies()

    def load_d2v_model(self, filepath):
        '''
        Loads pre-trained Doc2Vec model saved under filename.
        Also gets movie vectors from model.

        Input:
        ------
        filepath: str of file path
        '''
        self.d2v_model = Doc2Vec.load('models/{}'.format(filename))
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
        u_similar_movies = self.convert_to_unicode(similar_movies)
        return u_similar_movies

    def convert_to_unicode(self, recs):
        '''
        Input:
        ------
        recs: list of tuples of movie recommendations and their similarity
            probabilities or cosine similarities

        Output:
        -------
        u_recs: list with movie title strings converted from ASCII to unicode
        '''
        u_recs = []   # new list of tuples where movie title strings
                                # will be in unicode
        for movie, probab in recs:
            u_movie = movie.decode('utf-8')
            u_recs.append((u_movie, probab))
        return u_recs

    def alt_recommend_movies(self, pos_movies=[], neg_movies=[], num_recs=10):
        '''
        Instead of Doc2Vec, takes the average of all the tag vectors created by
        Word2Vec for each movie to get a vector representation of the movie.
        Then, add/subtracts those vectors to get a vector representation of the
        output. Then finds cosine similarity to all other movies, and returns
        the top movies that are closest.

        Input:
        ------
        pos_movies: list of indices of positive tags (tags to add)
        neg_tags: list of indices of negative tags (tags to subtract)
        num_recs: int of num recommendations desired (default is 10)

        Output:
        -------
        alt_similar_movies: list of tuples of similar movies and their cosine
        distances.
        '''
        #self.vectorize_movies()
        result_vector = self.get_result_vector(pos_movies, neg_movies)
        sorted_indices, cosine_similarities = self.compute_cosine(result_vector)
        alt_similar_movies = []
        i = 0
        while len(alt_similar_movies) < num_recs:
            # If similar movie is same as movies entered, skip over it
            if sorted_indices[i] in pos_movies or sorted_indices[i] in neg_movies:
                i += 1
            else:
                alt_similar_movies.append((self.movie_df.title.loc[sorted_indices[i]], \
                    cosine_similarities[i]))
                i += 1
        u_alt_similar_movies = self.convert_to_unicode(alt_similar_movies)
        return u_alt_similar_movies

    def get_result_vector(self, pos_movies, neg_movies):
        '''
        Takes in lists of indices of positive movies and negative movies.
        For each movie, calculates vector by averaging vectors of tags for movie.
        Adds pos_vectors and subtracts neg_vectors to get result vector, returns
        result vector.
        '''
        #one_tag = self.movie_df.tags.loc[pos_movies[0]][0] # Get an example tag
        tag_vec_size = len(self.tag_vectors['Action']) # Get size of a tag vector
        pos_vectors = np.zeros([len(pos_movies),tag_vec_size])
        for idx, movie_idx in enumerate(pos_movies):
            pos_vectors[idx] = self.movie_vector_matrix[movie_idx]
        if len(neg_movies) != 0:
            neg_vectors = np.zeros([len(neg_movies),tag_vec_size])
            for idx, movie_idx in enumerate(neg_movies):
                neg_vectors[idx] = self.movie_vector_matrix[movie_idx]
        else:
            neg_vectors = np.zeros([1,tag_vec_size])
        result_vector = np.sum(pos_vectors, axis=0) - np.sum(neg_vectors, axis=0)
        return result_vector

    def vectorize_movies(self):
        '''
        Creates vector representations of all movies in dataframe based on their
        tags. Initializes movie_vector_matrix: numpy array with number of rows
        being number of movies, and number of columns being size of tag vectors.
        '''
        tag_vec_size = len(self.tag_vectors['Action']) # Get size of a tag vector
        self.movie_vector_matrix = np.empty([self.movie_df.shape[0],tag_vec_size])
        print '\nVectorizing all movies...'
        for movie_idx in xrange(self.movie_df.shape[0]):
            tags_lst = self.movie_df.tags.loc[movie_idx]
            num_tags = self.movie_df.tags_length.loc[movie_idx]
            tag_vec_array = np.empty([num_tags, tag_vec_size])
            for i in xrange(num_tags):
                try:
                    tag_vec_array[i] = self.tag_vectors[tags_lst[i]]
                except KeyError:
                    tag_vec_array[i] = np.zeros([1,tag_vec_size])
            movie_vector = np.mean(tag_vec_array, axis=0)
            self.movie_vector_matrix[movie_idx] = movie_vector
        print 'Finished creating movie vector matrix.\n'

    def compute_cosine(self, result_vector):
        '''
        Input:
        ------
        result_vector: numpy array to compare with each and every movie vector
        to calculate the cosine similarities between them.

        Output:
        -------
        cosine_similarities: sorted numpy array of movie indices and the
        corresponding cosine similarity; descending (so closest movies should show
        up first)
        '''
        cosine_similarities = np.zeros(self.movie_df.shape[0])
        for i in xrange(len(cosine_similarities)):
            cosine_similarities[i] = 1 - cosine(result_vector, self.movie_vector_matrix[i])
        sorted_indices = np.argsort(cosine_similarities)[::-1] # Get sorted movie indices
        cosine_similarities = cosine_similarities[sorted_indices] # Sort cosine similarities
        return sorted_indices, cosine_similarities

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
    pass
