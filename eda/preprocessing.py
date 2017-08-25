# Movie2Vec
# Preprocessing
# Modifying df to make it palatable to word2vec model

import pandas as pd

def clean_columns(imdb_df):
    '''
    Selects & cleans columns in IMDB df.
    '''
    # Select pertinent columns from imdb_df
    movie_df = imdb_df[['movie_title','director_name','actor_1_name', \
        'actor_2_name','actor_3_name','genres','plot_keywords', 'country', \
        'content_rating']]

    # # Only retain movies with directors & years listed - movies without directors
    # # seem to be TV shows, not movies, only 4 rows w/o year
    # movie_df = movie_df[movie_df['director_name'].notnull()]
    # movie_df = movie_df[movie_df['title_year'].notnull()]
    # movie_df = movie_df[movie_df['country'].notnull()]

    # Replace null values in columns with empty lists (since we'll be adding
    # lists together later to get one list of all tags)
    for column in movie_df.columns:
        if column == 'plot_keywords': # Since we'll be splitting this string up later
            movie_df[column] = movie_df[column].fillna(value='')
        else:
            movie_df.loc[movie_df[column].isnull(), [column]] = \
                movie_df.loc[movie_df[column].isnull(), column].apply(lambda x: [])

    # Split strings in genres & plot_keywords columns
    movie_df.genres = movie_df.genres.apply(lambda x: x.split('|'))
    movie_df.plot_keywords = movie_df.plot_keywords.apply(lambda x: x.split('|'))

    # Put values in lists so that they can be added in create_tags function
    for column in movie_df.columns:
        if column != 'movie_title':
            movie_df[column] = movie_df[column].apply(lambda x: listify(x))

    return movie_df

def listify(x):
    bad_list = ['']
    empty_list = []
    if type(x) == list:
        if x != bad_list:
            return x
        else:
            return empty_list
    else:
        return [x]

def create_tags(movie_df):
    '''
    Input: df where first column is movie title, and every subsequent column
    is a feature that needs to be converted to a tag. Each value in these
    columns is a list.

    Output: df with 2 columns, one for movie title, second for list of tags
    (strings) corresponding to that movie.
    '''
    movie_df['tags'] = movie_df.director_name + movie_df.actor_1_name + \
        movie_df.actor_2_name + movie_df.actor_3_name + movie_df.genres + \
        movie_df.plot_keywords + movie_df.content_rating + movie_df.country
    # Drop all columns that are not movie_title or tags
    movie_df.drop(movie_df.columns[[1,2,3,4,5,6,7,8]], axis=1, inplace=True)
    return movie_df

def preprocess(imdb_df):
    '''
    Input: IMDB movie dataframe that's been loaded from csv.

    Output: Movie dataframe with one column for movie title and another column
    for movie tags - a list of strings such as movie director, genres, plot
    keywords, etc.
    '''
    cleaned_movie_df = clean_columns(imdb_df)
    movie_df = create_tags(cleaned_movie_df)

    movie_df.tags = movie_df.tags.apply(lambda x: replace_space(x))
    # Clean up movie title string
    movie_df.movie_title = movie_df.movie_title.apply(lambda x: x.strip('\xc2\xa0'))
    # Reset indices so no missing indices, just in case
    movie_df = movie_df.reset_index(drop=True)
    return movie_df

def replace_space(tags):
    '''
    Replaces spaces with underscores for the tags in the movies.tags column
    of movies dataframe
    '''
    new_tags = []
    for tag in tags:
        tag = tag.replace(' ','_')
        new_tags.append(tag)
    return new_tags

if __name__ == '__main__':
    # Get data
    filepath = '../data/imdb5000.csv'
    imdb_df = pd.read_csv(filepath)

    # Process dataframe
    movie_df = preprocess(imdb_df)
