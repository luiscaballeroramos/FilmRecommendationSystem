# FilmRecommendationSystem
 Final Project (Machinelearning & Python Course):FilmRecommendationSystem

The analysis has 3 parts:
1) Preprocessing
	The data from "DataSet_movielens_100k" folder is loaded
	movies and ratings are composing a pivot matrix called movie_ratings
2) Functions
	generates auxiliary functions in order to get following variables:
	MoviesView=times that each movie has been watched
	MoviesPopularity=mean of the ratings each movie has
	Plot_PopularityVSViews=graphic of Popularity(Views)
	this variables can be defined for each movie genre available
3) Postprocessing
	generates a Loop that allows 2 actions:
	Plot_PopularityVSViews: to see the relationship between this two variables
	kNN Algorithm: to see if its posible to predict the popularity from the views and the genre
	result is printed to analyze the accuracy of kNN in each genre