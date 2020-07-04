# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:19:59 2020

Prepare my .csv type pandas dataframes for export to SQL database (postgres)

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



# ## ----- Add urls to df_rec dataset: -----
# ## Get urls and servings (scraped at a later point in time)
# with open('epi_recipes_detailed', 'r') as io:
#         data = json.load(io)
# df_urls = pd.DataFrame(data)

# ## get previously preprocessed data frame
# df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_clean.csv',
# 					 index_col=0)

# ## Merge the two
# df_urls = df_urls.loc[:,['url', 'servings']]
# df_urls['index'] = list(df_urls.index)

# test = df_rec.merge(df_urls, how='left', left_on='index', right_on='index')

# ## verify that output is correct!
# for i, (title, url) in enumerate(zip(test['title'], test['url'])):
# 	if i % 100 == 0:
# 		print(title, url)

# ## Update df_rec and save to csv file
# df_rec = test
# df_rec.to_csv(r'D:\data science\nutrition\epi_recipes_clean.csv')




# ##########
# ## Do the same for a later processing step:
# df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_with_ghg.csv', 
# 					 index_col=0)

# ## Merge 
# test = df_rec.merge(df_urls, how='left', left_on='index', right_on='index')

# ## verify that output is correct!
# for i, (title, url) in enumerate(zip(test['title'], test['url'])):
# 	if i % 100 == 0:
# 		print(title, url)

# ## Update and save to csv file
# df_rec = test

# change servings column to strings
#df_rec.to_csv(r'D:\data science\nutrition\epi_recipes_with_ghg.csv')



## Finally, postgres SQL does not like line-breaks where there shouldn't be 
## any, and my different text columns probably contain a bunch. So loop through
## them and delete line-breaks with '\r'




## SQL cannot handly line breaks other than the ones inbetween rows, so let's
## write a function that takes a pandas.Series with strings as input and re-
## moves all line breaks
ds = df_rec['ingredients']
def remove_linebreaks(ds):
	'''
	Parameters
	----------
	ds : pandas.Series or list
		contains strings 

	Returns
	-------
	l : list
		contains strings without linebreaks
	'''
	l = []
	for i in ds:
		l.append(i.replace("\n", " ").replace("\r", " "))
	return l


df = df_rec
df['ingredients'] = remove_linebreaks(df_rec['ingredients'])
df['directions'] = remove_linebreaks(df_rec['directions'])
df['categories'] = remove_linebreaks(df_rec['categories'])
# TODO process desc (contains NaNs)
df['desc'] = remove_linebreaks(df_rec['desc'])
df['url'] = remove_linebreaks(df_rec['url'])
# TODO process servings (is list now)
#df['servings'] = remove_linebreaks(df_rec['servings'])



df.to_csv(r'D:\data science\nutrition\data\epi_recipes_with_ghg.csv')



## Save only columns where line-breaks really shouldn't be a problem for now
df_basic = df.loc[:,['index', 'title', 'date', 'rating', 'calories', 'sodium', 
				   'fat', 'protein', 'ghg', 'prop_ing', 
				   'ghg_log10', 'url']]
df_basic.to_csv(r'D:\data science\nutrition\data\recipes_sql.csv')





## Prepare ingredients dataframe

# Load preprocessed ingredient data (from impute_ingredient_ghg_values.py)
df_ing = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv',
					 index_col=1)
print(df_ing.columns)



# Some recipe_ids are strange (e.g. 1/3), actually the whole data for those 
# rows is messed up (repeating 1/3 in each column...), there are only 23 in 
# total, so for now just drop them:
fail_idx = []
failed_recids = []
recid_list = []
for i, (ipt, rid) in enumerate(zip(df_ing['input'], df_ing['recipe_id'])):
	if '/' in str(rid):
		fail_idx.append(i)
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
		
df_ing = df_ing.drop(df_ing.index[fail_idx])

# Drop all rows that do not exist in df_rec (wouldn't be able to reach them
# anyway, and missing foreign keys are not allowed in sql)
ing_recipe_ids = df_ing['recipe_id'].unique()
print('Unique recipe IDs in recipes table:', len(df_rec['index']))
print('Unique recipe IDs in ingredients table:', len(ing_recipe_ids))

(list(set(ing_recipe_ids) - set(df_rec['index'].values))) 

rec_to_ing = []
ing_to_rec = []
for i in ing_recipe_ids:
	if i not in df_rec['index']:
		ing_to_rec.append(i)
		
for i in df_rec['index']:
	if i not in ing_recipe_ids:
		rec_to_ing.append(i)
		
print('Recipe IDs in ingredients table not found in recipes table:', ing_to_rec)
print('Recipe IDs in recipes table not found in ingredients table:', rec_to_ing)


np.setdiff1d(list(ing_recipe_ids), list(df_rec['index'].values))
	
# Keep only columns for 
# ingredientsID is implied by the index (saved automatically)
# recipe_id = recipeID
# ghg = emissions
# ghg_missing = emissions_missing
# ghg_new = emissions_imputed
df_ing.rename(columns={"Unnamed: 0": "ingredientsID"}, inplace=True)
df_ing_basic = df_ing.loc[:,['ingredientsID', 'recipe_id', 'ghg', 'ghg_missing', 'ghg_new']]
df_ing_basic.to_csv(r'D:\data science\nutrition\data\ingredients_sql.csv', index=False)













