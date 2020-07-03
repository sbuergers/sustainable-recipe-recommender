# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:41:09 2020

@author: sbuer
"""


# Allow OS manipulations (e.g. deleting files)
import os

# Add project folder to search path
import sys
sys.path.append(r'D:\data science\nutrition\scripts\tdi_challenge_may2020')

# pandas
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

# numpy
import numpy as np

# Data management
import pickle
import json

# Check execution time
import time

# Regular expressions
import re

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Handle sparse matrices efficiently
import scipy
from scipy.sparse import csr_matrix

# Quantify similarity between recipes
from sklearn.metrics.pairwise import cosine_similarity





## I should probably plug these two into a SQL databse...

# Load recipe data (note that the row index matches the df_ing, while the
# column index matches to df_dummy)
df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_clean.csv',
					 index_col=0)

# Load preprocessed ingredient data (from impute_ingredient_ghg_values.py)
df_ing = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv',
					 index_col=0)

# Make sure all recipe ids are seen as strings:
recipe_id_str = [str(df_ing.loc[idx,'recipe_id']) for idx in df_ing.index]
df_ing['recipe_id'] = recipe_id_str



# I seem to have messed up a few recipe_id entries, try to fix
failed_recids = []
recid_list = []
for i, (ipt, rid) in enumerate(zip(df_ing['input'], df_ing['recipe_id'])):
	if '/' in str(rid):
		failed_recids.append(rid)
		# try to match by ingredients in df_rec and input in df_ing
		for idx in df_rec.index:
			ingre = df_rec.loc[idx,'ingredients'].split(',')[0].replace("[", "").replace("'", "").replace("]", "").lower()
			if ingre == ipt:
				print('Match foun', idx, ingre, ipt)
				recid_list.append(int(idx))
				break
	else:
		recid_list.append(int(rid))
		
df_ing['recipe_id_new'] = recid_list


## Check that they actually match by recipe_id in df_ing:
randIdx = np.random.randint(low=0, high=df_rec.shape[0], size=1)[0]
print(df_ing.loc[df_ing['recipe_id'] == str(randIdx), 'input'])
print(df_rec.loc[randIdx,'ingredients'])





## Estimate GHG emissions for each recipe
not_found = []
recipe_ghg = []
prop_ing_matched = [] # proportion of ingredients with GHG estimates
for i, (idx, title) in enumerate(zip(df_rec.index, df_rec['title'])):
	
	if idx in df_ing['recipe_id']:
		tmp = df_ing.loc[df_ing['recipe_id']==str(idx),'ghg_new']
		ghg_est = np.sum(tmp)
		prop_matched = np.sum(tmp>0)/len(tmp)
	else:
		ghg_est = 0
		not_found.append(idx)

	if i % 100 == 0:
		print('Recipe', i, '-->', title, '-->', ghg_est)

	recipe_ghg.append(ghg_est)
	prop_ing_matched.append(prop_matched)
	
	
	
## Add estimates to df_rec
df_rec['ghg'] = recipe_ghg
df_rec['prop_ing'] = prop_ing_matched
df_rec['ghg_log10'] = recipe_ghg
df_rec.loc[df_rec['ghg_log10']>0, 'ghg_log10'] = np.log10(df_rec['ghg_log10'][df_rec['ghg_log10']>0])





## Save df_rec with ghg emission columns 
df_rec.to_csv(r'D:\data science\nutrition\epi_recipes_with_ghg.csv')





	
## ----- Visualize GHG emissions -----

## Reload recipe data
df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_with_ghg.csv',
					 index_col=0)





## Recipe GHG emission histogram
plt.hist(df_rec['ghg'].dropna(), bins=300)
plt.suptitle('Estimated GHG emission distribution from 35000 recipes')
plt.xlim(0, 100)
plt.xlabel('Green house gas emissions (kg CO2)')
plt.ylabel('Number of recipes')
plt.savefig('Figures/recipe_ghg_histogram_ingredients.png', dpi=300)
plt.show()


## A log-ghg scale should be easier to visualize
sns.distplot(df_rec['ghg_log10'].dropna())
plt.suptitle('Estimated GHG emission distribution from 35000 recipes')


## There is a conspicuous peak at around 1.4 (25), why?
df_rec[(df_rec['ghg_log10']>1.35) & (df_rec['ghg_log10']<1.45)]



## Illustrate GHG emissions for the base ingredient categories:
## Read in look-up table for GHG emissions / sustainability
df_GHG = pd.read_csv('GHG-emissions-by-life-cycle-stage.csv')

## Visualize total GHG emissions
srtid = df_GHG['Total'].sort_values().index
plt.rcdefaults()
ax = df_GHG.loc[srtid, 'Land use change':'Retail'].plot(kind='barh', stacked=True, figsize=(10,8))
ax.set_yticklabels(df_GHG['Food product'][srtid])
ax.set_xlabel('kg-CO2 emissions by kg-product')
ax.set_title('Green house gas emissions by food product')
plt.show()



# # Replot with fewer ingredients fo illustration:
# label_list = ['Beef (beef herd)', 'Pig Meat', 'Cheese', 'Dark Chocolate', 
# 			  'Coffee', 'Olive Oil', 'Eggs', 'Tofu', 'Milk', 'Wheat & Rye (Bread)',
# 			  'Peas', 'Bananas', 'Apples']
# df_GHG.set_index(df_GHG['Food product'], inplace=True)

# srtid = df_GHG.loc[label_list, 'Total'].sort_values().index

# df_GHG = df_GHG.loc[label_list, :]
# plt.rcdefaults()
# ax = df_GHG.loc[srtid, 'Land use change':'Retail'].plot(kind='barh', stacked=True, figsize=(6,4))
# ax.set_yticklabels(df_GHG['Food product'][srtid])
# ax.set_xlabel('kg-CO2 emissions by kg-product')
# ax.set_title('Green house gas emissions by food product')
# plt.savefig('Figures/base_category_GHG_emissions.png', dpi=300,bbox_inches='tight')
# plt.show()






## ----- Try to recommend some recipes (content based filtering) ------


# Load category dummy coded data 
df_dummy = pd.read_csv(r'D:\data science\nutrition\epi_recipes', index_col=0)

# Remove dummy rows where recipe ingredients are anyway missing
df_dummy = df_dummy.loc[df_dummy.index.isin(df_rec['index'])]


# (re-)load sparse user recipe matrix
# NOTE: I should be able to speed this up if I only save the jagged array
# parts of the csr matrix 
try:
	SM = scipy.sparse.load_npz(r'D:\data science\nutrition\content_category_similarity.npz')
except:
	# Compute cosine similarity matrix (takes ~2 min to run)
	SM = cosine_similarity(csr_matrix(df_dummy.to_numpy(dtype=np.int8)), 
						dense_output=False)
	# takes ~10 min to run (2.4 GB)
	scipy.sparse.save_npz('content_category_similarity.npz', SM)
	
	
	
## --- Figures ---

# Compute a smaller version to be able to plot 
sm = cosine_similarity(csr_matrix(df_dummy.to_numpy()[0:100,:]))

# Show heatmap
sns.heatmap(sm[0:10,0:10], annot=True)

# Show heatmap with labels
df_hm = pd.DataFrame(sm)
df_hm.set_index(df_rec['title'][0:len(df_hm)], inplace=True)
df_hm.columns = list(df_rec['title'][0:len(df_hm)])
sns.heatmap(df_hm.iloc[0:10,0:10], annot=True)

# And a few more entries:
ax = sns.heatmap(df_hm.iloc[0:20, 0:20])
ax.xaxis.set_ticks_position('top')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()



def find_related_recipes(name, df_rec, N, SM):
	rec_id = df_rec.index[df_rec['title'] == name]
	similarities = np.flip(np.sort(SM[rec_id].todense(),axis=1)[:,-N:])
	similarities = np.squeeze(np.asarray(similarities))
	rel_rec_ids = np.flip(np.argsort(SM[rec_id].todense(),axis=1)[:,-N:])
	rel_rec_ids = np.squeeze(np.asarray(rel_rec_ids))
	related_recipes = df_rec.loc[rel_rec_ids,:]
	return(related_recipes, similarities)

	
def find_related_sustainable_recipes(name, df_rec, N, SM):
	rec_id = df_rec.index[df_rec['title'] == name]
	similarities = np.flip(np.sort(SM[rec_id].todense(),axis=1)[:,-N:])
	similarities = np.squeeze(np.asarray(similarities))
	rel_rec_ids = np.flip(np.argsort(SM[rec_id].todense(),axis=1)[:,-N:])
	rel_rec_ids = np.squeeze(np.asarray(rel_rec_ids))
	ghg_em = df_rec.iloc[rel_rec_ids,:]['ghg']
	# combine similarities and sustainabilities
	# compute ghg emissions of each recipe relative to the searched recipe
	ghg_em_norm = ghg_em / df_rec[df_rec['title']==name]['ghg'].values
	comb_score = ghg_em_norm + (1-similarities)
	rel_rec_ids = rel_rec_ids[np.argsort(comb_score)]
	related_recipes = df_rec.iloc[rel_rec_ids,:]
	return(related_recipes, similarities)


# Helper __function to print out recipe information__ including ingredients and tags


def show_recipe_ingredients(name, df_rec):
    df_recipe = df_rec.loc[df_rec['title'] == name, :]
    df_recipe = df_recipe.loc[:, (df_recipe !=0).any(axis=0)]
    N = len(df_recipe.columns)
    print('----------------------------------------------------------------')
    print('Name: ', df_recipe['title'].values)
    print('................................................................')
    print('Rating: ', df_recipe['rating'].values)
    print('Calories: ', df_recipe['calories'].values)
    print('Protein: ', df_recipe['protein'].values)
    print('Fat: ', df_recipe['fat'].values)
    print('Sodium: ', df_recipe['sodium'].values)
    print('Estimated GHG emissions: ', df_recipe['ghg'].values)
    print('Proportion ingredients with GHG estimate: ', df_recipe['prop_ing'].values)
    print('Number of ingredients or tags: ', df_recipe.shape[0])
    print('Tags and ingredients: ')
    for i, tag in enumerate(df_recipe.columns.tolist()[6:(N-3)]):
        print(i, tag)
    print(' ')


# __Find related recipes to a random recipe__ and print ingredient and tag information 


import random

recipe = random.choice(df_rec['title'])
print(recipe)
N_rel_rec = 10
rel_rec, sim = find_related_recipes(recipe, df_rec, N_rel_rec, SM)


# Show output (input recipe, and respective output recipes)

print('FIND ALTERNATIVES FOR RECIPE:')
show_recipe_ingredients(recipe, df_rec)
## skip first, because it again shows the recipe itself
for i, sim_recipe in enumerate(rel_rec['title'][1:]):
    print(' ')
    print('SIMILAR RECIPES:', i+1)
    print('Cosine similarity:', sim[i+1])
    show_recipe_ingredients(sim_recipe, df_rec)


print('FIND SUSTAINABLE ALTERNATIVES FOR RECIPE:')
show_recipe_ingredients(recipe, df_rec)
## skip first, because it again shows the recipe itself
for i, sim_recipe in enumerate(rel_rec['title'][1:]):
    print(' ')
    print('SIMILAR RECIPES:', i)
    print('Cosine similarity:', sim[i])
    show_recipe_ingredients(sim_recipe, df_rec)








## ------- SVD++ -------

# Load rating dataframe (user | title | rating)

# Helper function to see all methods of an object
def get_methods(obj):
	return [meth for meth in dir(obj) if callable(getattr(obj, meth))]


# loop handling
import itertools

# Surprise libraries
from surprise import Dataset
from surprise import Reader

# svd++ for recommender systems
from surprise import SVDpp

# model training, testing and hyperparameter grid search
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

# writing data to file
from surprise import dump

from collections import defaultdict



## Load user reviews 
df_users = pd.read_csv(r'D:\data science\nutrition\epi_reviews_75plus_w_usernames.csv', index_col=0)
df_users = df_users.loc[:,'user':'rating']

# formalize rating scale
reader = Reader(rating_scale=(1, 4)) # for centered: (-3, 3)

# put data into surprise format
data = Dataset.load_from_df(df_users, reader)
print(get_methods(data))


# Do a Grid Search for different hyperparameter values (earlier I tried this
# using only users with at least 8 ratings and the defaults were best for
# n_epochs, lr_all and reg_all, so I will fix them here and vary n_factors):
	
# # Note that the handbook suggests using different lrs for different params
# param_grid = {'n_factors': [10, 15, 20]}
# gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=5)

# gs.fit(data)

# # best RMSE score
# print(gs.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])

# # all cross validation results from grid
# gs.cv_results

# Also for n_factors the default value of 20 performs best




# Fit a default SVD++ model using all training data and check predictions
# (see https://surprise.readthedocs.io/en/stable/getting_started.html)
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

## Check precision and recall for K predictions for each user
kf = KFold(n_splits=2)
algo = SVDpp()

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=15, threshold=3.5)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))
	
## An item is considered relevant if its true rating rui is greater than a 
## given threshold. An item is considered recommended if its estimated rating 
## r^ui is greater than the threshold, and if it is among the k highest 
## estimated ratings.

## Remember that:
## Recall = Sensitivity = TP / (TP + FN)
## Recall = |{Recommended items that are relevant}| / |{Relevant items}|
## Precision = TP / (TP + FP)
## Precision = |{Recommended items that are relevant}| / |{Recommended items}|
	

# This creates a full "trainset", using all the data
trainset = data.build_full_trainset()

# Fit model
algo = SVDpp()
algo.fit(trainset)

# Save algorithm to file
filename = r'D:\data science\nutrition\collab_filter_algo'
dump.dump(filename, algo=algo)

# Reload 
_, algo = dump.load(filename)



# Show distribution of ratings by users
df_users['user'].value_counts()
df_users['title'].value_counts()
df_users[df_users['user']=='lschmidt']

# For a given user and recipe, compare true rating with predicted rating
uid = 'lschmidt'  
iid = 'acorn-squash-with-kale-and-sausage-51203850'
r = float(df_users.loc[(df_users['user']==uid) & (df_users['title']==iid),'rating'].values)

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=r, verbose=True)


# I can try this for all recipes this user liked
def show_user_predictions(uid, df, algo):
	rated_recipes = df.loc[df['user']==uid, 'title'].values
	for iid in rated_recipes:
		r = float(df.loc[(df['user']==uid) & (df['title']==iid),'rating'].values)
		pred = algo.predict(uid, iid, r_ui=r, verbose=True)
		print(pred)
		
		
show_user_predictions('lschmidt', df_users, algo)




# Get the top n predictions for each user
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
        [(raw item id, rating estimation, actual rating), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# predict ratings for all pairs (u, i) that are NOT in the training set.
# I cannot do this for the whole dataset (memory runs out rather quickly),
# but I can use the 10 users with the most ratings for example:
top10_raters = df_users['user'].value_counts().index[0:10].values
df10 = df_users[df_users['user'].isin(top10_raters)].copy()
df10.reset_index(inplace=True, drop=True)


data10 = Dataset.load_from_df(df10, reader)
top10_trainset = data10.build_full_trainset()
top10_testset = top10_trainset.build_anti_testset()
predictions = algo.test(top10_testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for i, (uid, user_ratings) in enumerate(top_n.items()):
	print(' ')
	print('User', i)
	print(uid)
	print('--------- liked ----------')
	dfu = df10[df10['user']==uid].copy()
	dfu.reset_index(inplace=True, drop=True)
	idx = np.flip(np.argsort(dfu['rating']))
	print(dfu[['title', 'rating']].loc[idx[0:20]])
	print(' ')
	print('------- predicted to like --------')
	[print(round(r*100)/100, '/', round(r_est*100)/100, iid) for (iid, r_est, r) in user_ratings]



# 





# eof