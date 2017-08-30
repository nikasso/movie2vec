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

# # Load movie dataframe
# with open('../data/ml-latest_df.pkl') as f:
#     movie_df = pickle.load(f)
# # Load 2vec models
# w2v_model = Word2Vec.load('../model/trial2_w2v_model')
# d2v_model = Doc2Vec.load('../model/trial2_d2v_model')

m2v = Movie2Vec()
#m2v.load_d2v_model('../model/models/d2v_model4')
m2v.load_w2v_model('../model/models/w2v_model2')


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
    data = str(request.form['movies_entered'])
    pos_movies = m2v.parse_input(data)
    #recs = m2v.recommend_movies(pos_movies=pos_movies) # Uses Doc2Vec
    recs = m2v.alt_recommend_movies(pos_movies=pos_movies) # Uses tag vector averaging
    return render_template('predict.html', movies_entered=data, recommendations=recs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
