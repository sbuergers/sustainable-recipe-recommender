# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:34:45 2020

@author: sbuer
"""


# Data science libraries
import numpy as np
import pandas as pd 

# Handle sparse matrices efficiently
import scipy
from scipy.sparse import csr_matrix

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Json
import json

# loop handling
import itertools





# Load rating dataframe (user | title | rating)
df_users = pd.read_csv('epi_users_reviews.csv')





# Surprise libraries
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from collections import defaultdict



# Only keep users with at least 10 rated items
v = df_users['user'].value_counts()
df_users_pruned = df_users[df_users['user'].isin(v.index[v.gt(9)])]
df_users_pruned.drop(columns=['Unnamed: 0'], inplace=True)


# Get dataframes in standard shape for surprise (uncentered and centered)
df = df_users_pruned.loc[:,'user':'rating']
df_c = df_users_pruned.drop(columns=['rating'])
df_c.columns = ['user', 'item', 'rating']

# assume unrated items are neutral, and rated items perfect scores
reader = Reader(rating_scale=(1, 4)) # for centered: (-3, 3)

# put data into surprise format
data = Dataset.load_from_df(df, reader)



# In[36]:

# define prediction algorithm
def set_algo(name="cosine", user_based=True, algo_type="KNNBasic"):
	'''Function to facilitate switching between different algorithms
    '''
		
	# To use item-based cosine similarity
	sim_options = {
	    "name": name,
	    "user_based": user_based, # Compute similarities between user or items
	}
	if algo_type=="KNNBasic":
		algo = KNNBasic(k=10, min_k=1, sim_options=sim_options)
		
	elif algo_type=="KNNWithMeans":
		algo = KNNWithMeans(k=10, min_k=1, sim_options=sim_options)
		
	elif algo_type=="KNNWithZScore":
		algo = KNNWithZScore(k=10, min_k=1, sim_options=sim_options)
		
	else:
		raise NameError('Unknown algorithm type.')
		
	return algo



# from https://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n




# Divide into training and test sets (there is also full cross validation)
# create training set
trainset, testset = train_test_split(data, test_size=.5)


# First train a KNN algorithm on training set.
algo = set_algo(name="cosine", user_based="True", algo_type="KNNWithMeans")
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)







top_n = get_top_n(predictions, n=2)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
	print(uid, [(iid, est) for (iid, est) in user_ratings])
	print(' ')
	print(' ')
	
	
	
	
	
## Using SVD (matrix factorization)
# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
	














import time
import random

import numpy as np
import six
from tabulate import tabulate

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import GridSearchCV



def compare_model_algorithms(data, Nrep=2, Nfolds=5):
	"""
	Prints out model performances and run times for standard algorithms in 
	Surprise.
	Input:
		data = surprise data object
		Nrep = number of iterations with different folds
		Nfolds = number of cross validation folds
	Output: 
		performance_list = list of performance matrices with 
			rows: RMSE, MAE, time(min); and
			cols: Algorithm (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans,
					KNNBaseline, CoClustering, BaselineOnly, NormalPredictor)
		performance = average over lists in performance_list
	"""
	
	# set RNG
	np.random.seed(0)
	random.seed(0)
	
	# set KNN algorithm options
	user_opt_cos = {"name":"cosine", "user_based":True}
	item_opt_cos = {"name":"cosine", "user_based":False}
	
	# The algorithms to cross-validate	
	s_SVD = SVD()
	s_SVDpp = SVDpp()
	s_NMF = NMF()
	s_SlopeOne = SlopeOne()
	u_KNNBasic = KNNBasic(sim_options=user_opt_cos)
	u_KNNWithMeans = KNNWithMeans(sim_options=user_opt_cos)
	u_KNNBaseline = KNNBaseline(sim_options=user_opt_cos)
	i_KNNBasic = KNNBasic(sim_options=item_opt_cos)
	i_KNNWithMeans = KNNWithMeans(sim_options=item_opt_cos)
	i_KNNBaseline = KNNBaseline(sim_options=item_opt_cos)
	s_CoClustering = CoClustering()
	s_BaselineOnly = BaselineOnly()
	s_NormalPredictor = NormalPredictor()
	
	classes = [s_SVD, s_SVDpp, s_NMF, s_SlopeOne, u_KNNBasic, u_KNNWithMeans, 
			u_KNNBaseline, i_KNNBasic, i_KNNWithMeans, i_KNNBaseline,
			s_CoClustering, s_BaselineOnly, s_NormalPredictor]
	
	class_names = ["SVD", "SVDpp", "NMF", "SlopeOne", "user-KNNBasic", "user-KNNWithMeans", 
				   "user-KNNBaseline", "item-KNNBasic", "item-KNNWithMeans", "item-KNNBaseline",
				   "CoClustering", "BaselineOnly", "NormalPredictor"]
	
	# repeat cross validation for different kfold splits for higher reliability
	performance_list = []
	headers = ['RMSE', 'MAE', 'Time (min)']
	for irep in range(0,Nrep):
		
		# cross validation folds will be the same for all algorithms. 
		kf = KFold(n_splits=Nfolds,random_state=0)  

		# cross validate for each algorithm
		table = np.zeros((len(classes),len(headers)))
		for ik, klass in enumerate(classes):
		    start = time.time()
		    out = cross_validate(klass, data, ['rmse', 'mae'], kf)
		    cv_time = (time.time() - start) / 60
		    mean_rmse = np.mean(out['test_rmse'])
		    mean_mae = np.mean(out['test_mae'])
		    table[ik,:] = np.array([mean_rmse, mean_mae, cv_time])
			
		# Accumulate results for each cross-validation	
		performance_list.append(table)
	
	# Show averaged results over cross validation iterations
	performance = sum(performance_list)/len(performance_list)
	print(tabulate(performance.tolist(), headers=headers, showindex=class_names))

	return performance_list, performance




# Run standard algorithms
perf_list, perf = compare_model_algorithms(data, Nrep=10, Nfolds=2)


# Compare to Baseline algorithm (expected rating for recipe)
perf_rel2_bl = perf - perf[11,:]
class_names = ["SVD", "SVDpp", "NMF", "SlopeOne", "user-KNNBasic", "user-KNNWithMeans", 
				   "user-KNNBaseline", "item-KNNBasic", "item-KNNWithMeans", "item-KNNBaseline",
				   "CoClustering", "BaselineOnly", "NormalPredictor"]
print(tabulate(perf_rel2_bl.tolist(), headers=headers, showindex=class_names))




## SVD++ seems to be doing the best. Let's see if we can improve on this
## further by using different parameters
algo = SVDpp

param_grid = {'n_epochs': [10, 20, 40], 
			  'lr_all': [0.001, 0.003, 0.005],
              'reg_all': [0.01, 0.02, 0.4, 0.6]}
gs = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# all cross validation results from grid
print(gs.cv_results)





# 




# fit model
algo.fit(trainset).test(testset)



# Use the full training set to make predictions!
# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo.fit(trainset)

# test algorithm on testset
predictions = algo.test(testset)





























# eof