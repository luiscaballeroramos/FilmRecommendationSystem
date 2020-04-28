# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:54:43 2020

@author: lcr
Final Project ML&Python
Recommendation
"""
import pandas as pd
import numpy as np
from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------
# PREPROCESSING
#-----------------------------------------------------------------------------
# DataSet
ratings_id = ['user_id', 'movie_id', 'rating','timestamp']  
ratings = pd.read_csv('DataSet_movielens_100k/ml-100k/u.data', sep='\t', names=ratings_id, usecols=range(3), encoding="ISO-8859-1")
movies_id = ['movie_id', 'movie_title','release_date','video_release_date','IMDb_URL',
             'Unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy',
             'Film-Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
movies = pd.read_csv('DataSet_movielens_100k/ml-100k/u.item', sep='|', names=movies_id, encoding="ISO-8859-1")
# Delete Columns Province/States; Lat; Long
movies =movies.loc[:, movies.columns != 'release_date']
movies =movies.loc[:, movies.columns != 'video_release_date']
movies =movies.loc[:, movies.columns != 'IMDb_URL']
# Ratings
ratings=pd.merge(movies,ratings)
# check error in rating(between 0 and 5)
ratings_error_rating=ratings.copy()
for i in range(0,5+1,1):
    ratings_error_rating=ratings_error_rating[ratings_error_rating['rating']!=i]
    pass
if len(ratings_error_rating)!=0:
    print('error in rating number, is not between limits')
    pass
# Movies Ratings
movies_ratings = ratings.pivot_table(index=['user_id'],columns=['movie_id'],values='rating')
# check error in movies_ratings (same number of ratings)
if len(ratings)!=movies_ratings.size-movies_ratings.isnull().sum().sum():
    print('error in movies_ratings, loss of information')
    pass
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

def Plot_PopularityVSViews(movies,movies_ratings,genre):
    # Plot of PopularityVSViews
    plt.scatter(MoviesViews(movies,movies_ratings,genre),MoviesPopularity(movies,movies_ratings,genre))
    plt.title(genre)
    plt.ylim(0,5)
    plt.xlim(0,MoviesViews(movies,movies_ratings,'All').max())
    plt.show()
    pass
#-----------------------------------------------------------------------------
# POSTPROCESSING
#-----------------------------------------------------------------------------
result={}
for genre in ['All','Unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']:
    # Plot of Movies Popularity VS movies Views
    Plot_PopularityVSViews(movies,movies_ratings,genre)
    # kNN Algorithm
    if len(MoviesViews(movies,movies_ratings,genre))>5:
        X=np.array(MoviesViews(movies,movies_ratings,genre),dtype=float)
        X=X.reshape(-1,1)
        y=np.array(MoviesPopularity(movies,movies_ratings,genre),dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        knn = neighbors.KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy=accuracy_score(y_test, y_pred)
        # # Confusion Matrix
        # from sklearn.metrics import confusion_matrix
        # print(confusion_matrix(y_test, y_pred))
        # # Classification Report
        # from sklearn.metrics import classification_report
        # print(classification_report(y_test, y_pred))
    else:
        accuracy=0
        pass
    result[genre]=accuracy
    pass
print(result)
