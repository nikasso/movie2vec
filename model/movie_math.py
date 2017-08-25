# Movie2Vec
# Movie Math
# Loads trained word2vec & doc2vec models so that similarity methods can be
# run on movies & directors by the Users

import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import sys
sys.path.insert(0, '/Users/nikhilmakaram/Documents/Python/Galvanize_DSI/movie2vec/eda')
import preprocessing

def do_some_math():
    print 'Most similar directors to: Steven Spielberg:'
    print tag_vectors.most_similar(positive=['Steven_Spielberg'])

    spielberg_abrams_sim = tag_vectors.similarity('J.J._Abrams','Steven_Spielberg')
    print 'Similarity between J.J. Abrams and Steven Spielberg:', \
        spielberg_abrams_sim

    print "The Matrix + On Her Majesty's Secret Service = "
    print movie_vectors.most_similar(positive=[654,3375])

def get_data():
    # Get data
    filepath = '../data/imdb5000.csv'
    imdb_df = pd.read_csv(filepath)
    # Prepare data to load into 2vec models
    movie_df = preprocessing.preprocess(imdb_df)
    return movie_df

if __name__ == '__main__':
    movie_df = get_data()

    w2v_model = Word2Vec.load('trial_w2v_model')
    d2v_model = Doc2Vec.load('trial_d2v_model')

    tag_vectors = w2v_model.wv
    movie_vectors = d2v_model.docvecs

    do_some_math()


'''
Examples:
---------
> tag_vectors.most_similar(positive=['Steven_Spielberg'])
>
[('Wolfgang_Petersen', 0.9462029933929443),
 ('Ridley_Scott', 0.9422283172607422),
 ('George_Lucas', 0.939114511013031),
 ('Terrence_Malick', 0.9237831830978394),
 ('Sam_Mendes', 0.9232810139656067),
 ('Robert_Zemeckis', 0.921073853969574),
 ('Marc_Forster', 0.9180954694747925),
 ('Alex_Proyas', 0.9180061221122742),
 ('Robert_Redford', 0.9177207946777344),
 ('Eamonn_Walker', 0.9169954657554626)]

> tag_vectors.similarity('J.J._Abrams','Steven_Spielberg')
> 0.83269052698220802

> movie_df.movie_title.loc[654]
> 'The Matrix'
> movie_df.movie_title.loc[3375]
> 'On Her Majesty's Secret Service'

> movie_vectors.most_similar(positive=[654,3375])
>
 [('Exit Wounds', 0.9083218574523926),
 ('Bottle Rocket', 0.9036020636558533),
 ('The Players Club', 0.9027529954910278),
 ('The Naked Gun 2\xc2\xbd: The Smell of Fear', 0.8981586694717407),
 ('Barbershop', 0.8967175483703613),
 ('Corky Romano', 0.8946095705032349),
 ('The Last Godfather', 0.8914022445678711),
 ('Hot Pursuit', 0.8901454210281372),
 ('Man of the House', 0.8899064660072327),
 ('The Campaign', 0.8872640132904053)]

 > movie_df.loc[movie_df.movie_title == 'The Godfather'].index[0]
 > 3466
 '''
