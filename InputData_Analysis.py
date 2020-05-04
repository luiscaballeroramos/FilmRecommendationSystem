# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:13:39 2020

@author: lcr
"""
import pandas as pd
from datetime import datetime
#-----------------------------------------------------------------------------
# PREPROCESSING
#-----------------------------------------------------------------------------
# Users
users_names=['user_id','age','gender','occupation','zip code']
users = pd.read_csv('DataSet_movielens_100k/ml-100k/u.user', sep='|',names=users_names, encoding="ISO-8859-1")
sers=users.set_index('user_id')
# Users' gender
users_genders=users[~users.duplicated('gender')]['gender'].tolist()
us# set index=movie_id (=>only and unique index)
uers_genders.sort()
# Users' ages
users_ages=users[~users.duplicated('age')]['age'].tolist()
users_ages.sort()
# Users' occupations
users_occupations=users[~users.duplicated('occupation')]['occupation'].tolist()
users_occupations.sort()

# Movies
movies_names = ['movie_id', 'movie_title','release_date','video_release_date','IMDb_URL',
             'Unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy',
             'Film-Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
movies = pd.read_csv('DataSet_movielens_100k/ml-100k/u.item', sep='|',names=movies_names, encoding="ISO-8859-1")
# set index=movie_id (=>only and unique index)
movies=movies.set_index(['movie_id'])

# Ratings
ratings_names = ['user_id', 'movie_id', 'rating','timestamp']
ratings = pd.read_csv('DataSet_movielens_100k/ml-100k/u.data', sep='\t', names=ratings_names, encoding="ISO-8859-1")
# Datetimes from Timestamp
ratings['datetime']=pd.to_datetime(ratings['timestamp'],unit='s')

# Pivot Table: user_id & movie_id => rating
user_movie_ratings=ratings.pivot_table('rating',index='user_id',columns='movie_id')


#-----------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------
def MoviesViews(movies,movies_ratings,genre):
    # Select Genre
    if genre!='All':
        movies=movies[movies[genre]==1]
        pass
    movies_id_genre=movies['movie_id'].tolist()
    # Movies Views
    movies_views=(~movies_ratings[movies_id_genre].isnull()).sum()
    return movies_views

def MoviesPopularity(movies,movies_ratings,genre):
    # Select Genre
    if genre!='All':
        movies=movies[movies[genre]==1]
        pass
    movies_id_genre=movies['movie_id'].tolist()
    # Movies Popularity=mean of ratings
    movies_popularity=movies_ratings[movies_id_genre].mean()
    return movies_popularity
