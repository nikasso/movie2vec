# Movie2Vec
# Model Trial 2
# 2nd trial models (Word2Vec & Doc2Vec) with MovieLens Latest data (has longer
# "sentences" per movies)

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import cPickle as pickle
import sys
sys.path.insert(0, '/Users/nikhilmakaram/Documents/Python/Galvanize_DSI/movie2vec/eda')
import preprocessing


def w2v():
    print 'Training Word2Vec model...'
    # Fit model to sentences
    w2v_model = Word2Vec(sentences, size=200, iter=20, window=max_sentence_size)
    print 'Word2Vec model trained.'
    w2v_model.save('trial2_w2v_model')
    tag_vectors = w2v_model.wv
    return w2v_model, tag_vectors

def d2v():
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        u_sentence = [unicode(x, 'utf-8') for x in sentence]
        tagged_sentences.append(TaggedDocument(u_sentence,[movie_df.title[i]]))
    print '\nTraining Doc2Vec model...'
    # Fit model to sentences
    d2v_model = Doc2Vec(tagged_sentences, dm=0, size=200, iter=20, \
        min_count=20, window=max_sentence_size)
    print 'Doc2Vec model trained.'
    d2v_model.save('trial2_d2v_model')
    movie_vectors = d2v_model.docvecs
    return d2v_model, movie_vectors


if __name__ == '__main__':
    # Load dataframe
    with open('../data/ml-latest_df.pkl') as f:
        movie_df = pickle.load(f)
    sentences = movie_df.tags.tolist()
    max_sentence_size = movie_df.tags_length.max() # 938
    w2v_model, tag_vectors = w2v()      # word2vec model & its tag vectors
    d2v_model, movie_vectors = d2v()    # doc2vec model & its movie vectors


'''
Hyperparameter Tuning Results
=============================
Word2Vec
--------
iter=50                             Not bad
iter=50, window=940                 Not great
iter=50, size=200, window=940
size=200
size=200, window=max_sentence_size  Not great

Doc2Vec
-------
iter=50                                 Not bad (pretty good for 12 Monkeys + Se7en)
iter=50, window=940                     Not great
iter=20, size=200, window=940           Not great
size=200                                Pretty good
size=200, window=max_sentence_size      Not great
dm=0, iter=20, window=max_sentence_size Pretty good (for 12 Monkeys + Se7en)

'''
