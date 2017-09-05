# Movie2Vec
# App
# Flask app for recommendations

from flask import Flask, render_template, request, jsonify
import cPickle as pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import sys
sys.path.insert(0, '/Users/nikhilmakaram/Documents/Python/Galvanize_DSI/movie2vec/model')
from movie2vec import Movie2Vec


app = Flask(__name__)

m2v = Movie2Vec()
m2v.load_w2v_model('../model/models/w2v_model4') # Choose Word2Vec model to load


@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can enter the
    movies they want to add."""
    return render_template('submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the movie(s) to add and recommend based off of from an input
    form and use the model to provide the recommendations.
    """
    pos_data = str(request.form['pos_movies_entered'])
    pos_movies, pos_tags = m2v.parse_input(pos_data)

    neg_data = str(request.form['neg_movies_entered'])
    neg_movies, neg_tags = m2v.parse_input(neg_data)

    # Uses Doc2Vec
    #recs = m2v.recommend_movies(pos_movies=pos_movies)

    # Uses tag vector averaging
    recs = m2v.alt_recommend_movies(pos_movies=pos_movies, pos_tags=pos_tags, \
        neg_movies=neg_movies, neg_tags=neg_tags)

    return render_template('predict.html', pos_movies_entered=pos_data, \
        neg_movies_entered=neg_data, recommendations=recs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
