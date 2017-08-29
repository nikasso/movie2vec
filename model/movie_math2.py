# Movie2Vec
# Movie Math 2 (For MovieLens Latest data)
# Loads trained word2vec & doc2vec models so that similarity methods can be
# run on movies & directors by the Users

import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import cPickle as pickle
import sys
sys.path.insert(0, '/Users/nikhilmakaram/Documents/Python/Galvanize_DSI/movie2vec/eda')
import preprocessing

def most_similar_tags():
    similar_tags = tag_vectors.most_similar(positive=['steven_spielberg'])
    print '\nMost similar tags to: Steven Spielberg'
    print '--------------------------------------'
    for tag in similar_tags[:5]:
        print tag

def tag_similarity():
    tag_similarity = tag_vectors.similarity('steven_spielberg','j.j._abrams')
    print '\nSimilarity between Steven Spielberg and J.J. Abrams'
    print '---------------------------------------------------'
    print tag_similarity

def most_similar_movies():
    similar_movies = movie_vectors.most_similar(positive=[2487,3543])
    print "\nThe Matrix + On Her Majesty's Secret Service = "
    print '-----------------------------------------------'
    for movie in similar_movies[:5]:
        print movie

    similar_movies2 = movie_vectors.most_similar(positive=[31,46])
    print "\nTwelve Monkeys + Se7en = "
    print '-----------------------------------------------'
    for movie in similar_movies2[:5]:
        print movie

    # similar_movies3 = movie_vectors.most_similar(positive=[8275,1202])
    # print "\nRay + Raging Bull = "
    # print '-----------------------------------------------'
    # for movie in similar_movies3[:5]:
    #     print movie

def do_some_math():
    '''
    Calls functions that calculate example similarities and print out results
    to terminal as examples for intuitive evaluation of model performances.
    '''
    print '\nMovie Math:'
    print '==================================================='
    most_similar_tags()
    tag_similarity()
    most_similar_movies()


if __name__ == '__main__':
    # Load dataframe
    with open('../data/ml-latest_df.pkl') as f:
        movie_df = pickle.load(f)
    # Load up models
    w2v_model = Word2Vec.load('trial2_w2v_model')
    d2v_model = Doc2Vec.load('trial2_d2v_model')
    # Load up tag & movie vectors
    tag_vectors = w2v_model.wv
    movie_vectors = d2v_model.docvecs

    do_some_math()


'''
Notes:
------
To get an index for a particular movie:
> movie_df.loc[movie_df.movie_title == 'The Godfather'].index[0]
> 3466

To get a movie at a particular index:
> movie_df.movie_title.loc[654]
> 'The Matrix'
'''
