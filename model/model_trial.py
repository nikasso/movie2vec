# Movie2Vec
# Model Trial
# Testing out creation of trial model with some preprocessed movie data (with
# a subset of features)

from __future__ import division
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import sys
sys.path.insert(0, '/Users/nikhilmakaram/Documents/Python/Galvanize_DSI/movie2vec/eda')
import preprocessing


def w2v():
    # Fit model to sentences
    w2v_model = Word2Vec(sentences, iter=50, sg=1)
    w2v_model.save('trial_w2v_model')
    tag_vectors = w2v_model.wv
    return w2v_model, tag_vectors

def d2v():
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        u_sentence = [unicode(x, 'utf-8') for x in sentence]
        tagged_sentences.append(TaggedDocument(u_sentence,[movie_df.movie_title[i]]))
    # Fit model to sentences
    d2v_model = Doc2Vec(tagged_sentences, iter=50)
    d2v_model.save('trial_d2v_model')
    movie_vectors = d2v_model.docvecs
    return d2v_model, movie_vectors

def get_data():
    # Get data
    filepath = '../data/imdb5000.csv'
    imdb_df = pd.read_csv(filepath)
    # Prepare data to load into 2vec models
    movie_df = preprocessing.preprocess(imdb_df)
    return movie_df

if __name__ == '__main__':
    movie_df = get_data()
    sentences = movie_df.tags.tolist()
    w2v_model, tag_vectors = w2v()      # word2vec model & its tag vectors
    d2v_model, movie_vectors = d2v()    # doc2vec model & its movie vectors
