# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:13:39 2020

@author: lcr
"""
import pandas as pd
import numpy as np

from Functions import (PreCleaning_Dataframe,
VbleType_Dataframe,
MissingData_Dataframe,MissingData_Delete_Dataframe,
DataConversion_Dataframe,DataConversion_Dataframe_FromString_ToDatetimeYear,DataConversion_Dataframe_FromTimestamp_ToDatetimeYear,
UniquenessTest,UniqueIndex,
EDA_Univariate_Dataframe,EDA_Multivariate_Dataframe)
#-----------------------------------------------------------------------------
# 00_RAW DATA
#-----------------------------------------------------------------------------
# Users
# features names
users_names=['user_id','age','gender','occupation','zip_code']
# load raw data
users = pd.read_csv('DataSet_movielens_100k/ml-100k/u.user', sep='|',names=users_names, encoding="ISO-8859-1")
#-----------------------------------------------------------------------------
# Movies
# features names
movies_names = ['movie_id', 'movie_title','release_date','video_release_date','IMDb_URL',
             'Unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy',
             'Film-Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
# load raw data
movies = pd.read_csv('DataSet_movielens_100k/ml-100k/u.item', sep='|',names=movies_names, encoding="ISO-8859-1")
#-----------------------------------------------------------------------------
# Ratings
# features names
ratings_names = ['user_id', 'movie_id', 'rating','timestamp']
# load raw data
ratings = pd.read_csv('DataSet_movielens_100k/ml-100k/u.data', sep='\t', names=ratings_names, encoding="ISO-8859-1")
#-----------------------------------------------------------------------------
# 01_CLEANING
#-----------------------------------------------------------------------------
# Users
PreCleaning_Dataframe(users)
# Data Identification
users=users.set_index(['user_id'])
# Unkown Data
users=users.drop(['zip_code'],axis=1)
# Variable Type
users_vbletype=VbleType_Dataframe(users,['numerical_discrete','categorical_nominal','categorical_nominal'])
# Missing Data
user_missingdata=MissingData_Dataframe(users,users_vbletype)
# Data Conversion
users,users_conversions=DataConversion_Dataframe(users,users_vbletype)
# Unique Values
#-----------------------------------------------------------------------------
# Movies
PreCleaning_Dataframe(movies)
# Data Identification
movies=movies.set_index(['movie_id'])
# Unkown Data
movies=movies.drop(['video_release_date'],axis=1)
movies=movies.drop(['IMDb_URL'],axis=1)
# Variable Type
movies_vbletype=VbleType_Dataframe(movies,['categorical_nominal',
                                'categorical_ordinal',
                                'categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal',
                                'categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal',
                                'categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal',
                                'categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal',
                                'categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal','categorical_nominal'])
# Missing Data
movies_missingdata=MissingData_Dataframe(movies,movies_vbletype)
movies=MissingData_Delete_Dataframe(movies,movies_missingdata)
# Data Conversion
movies,movies_conversions=DataConversion_Dataframe(movies,movies_vbletype)
movies=DataConversion_Dataframe_FromString_ToDatetimeYear(movies,'release_date')
# Uniqueness test
repeated_title=UniquenessTest(movies,'movie_title')
# Unique Values
movies,movies_movie_title_unique_index=UniqueIndex(movies,'movie_title',repeated_title,check_similarity=True)
#-----------------------------------------------------------------------------
# Ratings
PreCleaning_Dataframe(ratings)
# Data Identification
# Unkown Data
# Variable Type
ratings_vbletype=VbleType_Dataframe(ratings,['numerical_discrete','numerical_discrete','numerical_continuous',
                                'categorical_ordinal'])
# Missing Data
ratings_missingdata=MissingData_Dataframe(ratings,ratings_vbletype)
# Data Conversion
ratings,ratings_conversions=DataConversion_Dataframe(ratings,ratings_vbletype)
ratings=DataConversion_Dataframe_FromTimestamp_ToDatetimeYear(ratings,'timestamp')
# Unique Values
#-----------------------------------------------------------------------------
# Single movie_title in ratings
for i in range(movies_movie_title_unique_index.shape[0]):
    repeated=movies_movie_title_unique_index.iloc[i,:]['repeated']
    single=movies_movie_title_unique_index.iloc[i,:]['single']
    ratings.loc[ratings['movie_id']==repeated,'movie_id']=single
    pass
# NO Missing movie_id in ratings
for index,row in movies_missingdata.iterrows():
    for i in ratings[ratings['movie_id']==index].index:
        ratings=ratings.drop(index=i)
        pass
    pass
#-----------------------------------------------------------------------------
# Ratings Knowledge about User
ratings_user_knowledge=ratings.copy()
users_index_list=users.index.values.tolist()
for user_id in users_index_list:
    for col in users:
        aux=users.loc[user_id,:][col]
        ratings_user_knowledge.loc[ratings_user_knowledge['user_id']==user_id,col]=aux
        pass
    pass
#-----------------------------------------------------------------------------
# Ratings Knowledge about Movie
ratings_movie_knowledge=ratings.copy()
movies_index_list=movies.index.values.tolist()
for movie_id in movies_index_list:
    for col in movies:
        aux=movies.loc[movie_id,:][col]
        ratings_movie_knowledge.loc[ratings_movie_knowledge['movie_id']==movie_id,col]=aux
        pass
    pass


# User_Movie_Rating
# Pivot Table: user_id & movie_id => rating
user_movie_ratings=ratings.pivot_table('rating',index='user_id',columns='movie_id')

#-----------------------------------------------------------------------------
# 02_EXPLORATORY DATA ANALYSIS
#-----------------------------------------------------------------------------
# Users
#-----------------------------------------------------------------------------
EDA_Univariate_Dataframe(users,users_vbletype)
EDA_Multivariate_Dataframe(users,hue=None)
#-----------------------------------------------------------------------------
# Movies
EDA_Univariate_Dataframe(movies,movies_vbletype)
EDA_Multivariate_Dataframe(movies,hue=None)
#-----------------------------------------------------------------------------
# Ratings
EDA_Univariate_Dataframe(ratings,ratings_vbletype)
EDA_Multivariate_Dataframe(ratings,hue=None)
#-----------------------------------------------------------------------------
# Ratings Knowledge about User
EDA_Multivariate_Dataframe(ratings_user_knowledge,hue=None)
#-----------------------------------------------------------------------------
# Ratings Knowledge about Movie
EDA_Multivariate_Dataframe(ratings_movie_knowledge,hue=None)

#-----------------------------------------------------------------------------
# 03_MODEL
#-----------------------------------------------------------------------------
from surprise import Reader,Dataset,dump
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.knns import KNNBaseline,KNNBasic,KNNWithMeans,KNNWithZScore
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import SVD,SVDpp,NMF
from surprise.model_selection.search import RandomizedSearchCV

# InputData
RS_ratings = ratings.drop(columns='timestamp')
RS_reader = Reader(name=None,
               line_format='user item rating',
               sep=',',
               rating_scale=(1, 5),
               skip_lines=0)
RS_data = Dataset.load_from_df(RS_ratings, RS_reader)

# Benchmark_Algorithm_Metric
benchmark = []
for algorithm in [BaselineOnly(),CoClustering(),KNNBaseline(),KNNBasic(),KNNWithMeans(),KNNWithZScore(),NMF(),NormalPredictor(),SlopeOne(),SVD(),SVDpp()]:
    # Perform cross validation
    results = cross_validate(algorithm, RS_data, measures=['rmse','mae','mse','fcp'], cv=5, verbose=True)
    # Results To Serie List
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    pass
# Results to Dataframe and .csv
pd.DataFrame(benchmark).set_index('Algorithm').sort_index().to_csv('Benchmark_Algorithms_Metrics/Benchmark_Algorithms_Metrics.csv',sep='|')

#RandomizedSearchCV for Best Estimator
RS_similarity_options={'name':['cosine','pearson','pearson_baseline'],#,'msd'
               'min_support':[0,1,5,10]}
RS_baseline_options={'method': ['als', 'sgd'],
                      'reg': [1, 2]}
# KNNBaseline
RS_KNNBaseline_parameters={'k':[2,5,10,20,40,80,160,320],
               'min_k':[1,2,5,10],
               'sim_options':RS_similarity_options,
               'bsl_options':RS_baseline_options}
RS_RandomSearch=RandomizedSearchCV(KNNBaseline,
                   RS_KNNBaseline_parameters,
                   n_iter=90,
                   measures=['rmse','mae'],#,'mse','fcp'
                   cv=3,
                   random_state=np.random.seed(0),
                   joblib_verbose=10)
RS_RandomSearch.fit(RS_data)
dump.dump('Algorithms/KNNBaseline_rmse', algo=RS_RandomSearch.best_estimator['rmse'])
dump.dump('Algorithms/KNNBaseline_mae', algo=RS_RandomSearch.best_estimator['mae'])
pd.DataFrame.from_dict(RS_RandomSearch.cv_results).sort_index().to_csv('Algorithms/KNNBaseline_measures.csv',sep='|')
tmp = pd.DataFrame.from_dict(RS_RandomSearch.cv_results)

# SVD
RS_SVD_parameters={'n_factors':[20,50,100,200],
               'n_epochs':[5,10,20,40],
               'biased':[True],
               'init_mean':[0],
               'init_std_dev':[0.1],
               'lr_all':[0.0025,0.005,0.01],
               'reg_all':[0.01,0.02,0.05]}
best_rmse=1000
best_mae=1000
dict_cv={}
for i in range(1,90+1):
    RS_RandomSearch=RandomizedSearchCV(SVD,
                       RS_SVD_parameters,
                       n_iter=1,
                       measures=['rmse','mae'],#,'mse','fcp'
                       cv=3,
                       random_state=np.random.seed(i),
                       joblib_verbose=10)
    RS_RandomSearch.fit(RS_data)
    dict_cv[i]=RS_RandomSearch.cv_results
    if RS_RandomSearch.best_score['rmse']<best_rmse:
        best_rmse=RS_RandomSearch.best_score['rmse']
        tmp=RS_RandomSearch.best_estimator['rmse']
        dump.dump('Algorithms/SVD_rmse', algo=RS_RandomSearch.best_estimator['rmse'])
        pass
    if RS_RandomSearch.best_score['mae']<best_mae:
        best_mae=RS_RandomSearch.best_score['mae']
        tmp=RS_RandomSearch.best_estimator['mae']
        dump.dump('Algorithms/SVD_mae', algo=RS_RandomSearch.best_estimator['mae'])
        pass
    pass
pd.DataFrame.from_dict(dict_cv).sort_index().to_csv('Algorithms/SVD_measures.csv',sep='|')
tmp = pd.DataFrame.from_dict(RS_RandomSearch.cv_results)
# SlopeOne
RS_CV=cross_validate(SlopeOne(),
                   RS_data,
                   measures=['rmse','mae'],#,'mse','fcp'
                   cv=3,
                   verbose=True)
dump.dump('Algorithms/SlopeOne_rmse', algo=RS_CV)
dump.dump('Algorithms/SlopeOne_mae', algo=RS_CV)
pd.DataFrame.from_dict(RS_CV).sort_index().to_csv('Algorithms/SlopeOne_measures.csv',sep='|')




