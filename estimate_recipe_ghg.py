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
sns.distplot(df_rec['ghg'].dropna())
plt.suptitle('Estimated GHG emission distribution from 35000 recipes')

## A log-ghg scale should be easier to visualize
sns.distplot(df_rec['ghg_log10'].dropna())
plt.suptitle('Estimated GHG emission distribution from 35000 recipes')


## There is a conspicuous peak at around 1.4 (25), why?
df_rec[(df_rec['ghg_log10']>1.35) & (df_rec['ghg_log10']<1.45)]








## ----- Try to recommend some recipes (content based filtering) ------


# Load category dummy coded data 
df_dummy = pd.read_csv(r'D:\data science\nutrition\epi_recipes', index_col=0)

# Remove dummy rows where recipe ingredients are anyway missing
df_dummy = df_dummy.loc[df_dummy.index.isin(df_rec['index'])]


# (re-)load sparse user recipe matrix
try:
	SM = scipy.sparse.load_npz('content_category_similarity.npz')
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
N_rel_rec = 32
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


















# eof