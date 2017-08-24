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
import eda, preprocessing


def w2v():
    w2v_model = Word2Vec(sentences, min_count=0, iter=100)
    w2v_model.save('trial_w2v_model')
    #w2v_model.Word2Vec.load('trial_w2v_model')
    # Get length of vocab
    vocab_len = len(w2v_model.wv.vocab)
    tag_vectors = w2v_model.wv
    return w2v_model, tag_vectors

def d2v():
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        # for word in sentence:
        #     u_word = unicode(word, 'utf-8')
        u_sentence = [unicode(x, 'utf-8') for x in sentence]
        tagged_sentences.append(TaggedDocument(u_sentence,[i]))
    #print tagged_sentences[:2]
    d2v_model = Doc2Vec(tagged_sentences, min_count=0, iter=100)
    d2v_model.save('trial_d2v_model')
    #d2v_model.Word2Vec.load('trial_d2v_model')
    movie_vectors = d2v_model.docvecs
    #movie_vectors.most_similar(positive=[0,1], topn=1) <-- For most similar
    return d2v_model, movie_vectors

if __name__ == '__main__':
    # Get data
    imdb_df = eda.imdb_eda()
    movies = preprocessing.preprocess(imdb_df)
    sentences = movies.tags.tolist()

    w2v_model, tag_vectors = w2v()
    d2v_model, movie_vectors = d2v()

    tag_vectors.most_similar(positive=['Christopher_Nolan'])
    '''
    Output:
    [('Patrick_Ryan_Sims', 0.8957345485687256),
     ('Paul_Michael_Glaser', 0.8727496266365051),
     ('Christopher_McQuarrie', 0.8700274229049683),
     ('Simon_West', 0.8696989417076111),
     ('Jon_Amiel', 0.866500198841095),
     ('D.J._Caruso', 0.8640592694282532),
     ('John_Herzfeld', 0.8544985055923462),
     ('Pete_Travis', 0.8528991341590881),
     ('Rob_Cohen', 0.8492453694343567),
     ('Andrew_Davis', 0.8471401929855347)]
    '''

    movie_vectors.most_similar(positive=[0,1]) # Avatar + Pirates 3
    # Find movie titles
    #movies.movie_title.loc[<index of movie>]
    '''
    Output:
    [(234, 0.94736248254776),       Knight and Day
     (1454, 0.9432020783424377),    Mirrors
     (2963, 0.9404279589653015),    Crank
     (375, 0.937872052192688),      Contact
     (28, 0.9261223077774048),      Battleship
     (1670, 0.9246368408203125),    Winnie the Pooh
     (4573, 0.9201890230178833),    History of the World: Part I
     (1055, 0.9179477691650391),    The Cable Guy
     (1979, 0.9167566299438477),    Michael Collins
     (2317, 0.9164246320724487)]    Passchendaele
    '''
