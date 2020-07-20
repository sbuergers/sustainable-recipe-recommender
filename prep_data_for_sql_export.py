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

# Helper functions to convert json to dummy data frame
from categories_to_dummy import sublists_to_binaries, sublist_uniques




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




## SQL cannot handle line breaks other than the ones inbetween rows, so let's
## write a function that takes a pandas.Series with strings as input and re-
## moves all line breaks
# ds = df_rec['ingredients']
# def remove_linebreaks(ds):
# 	'''
# 	Parameters
# 	----------
# 	ds : pandas.Series or list
# 		contains strings 

# 	Returns
# 	-------
# 	l : list
# 		contains strings without linebreaks
# 	'''
# 	l = []
# 	for i in ds:
# 		l.append(i.replace("\n", " ").replace("\r", " "))
# 	return l


# df = df_rec
# df['ingredients'] = remove_linebreaks(df_rec['ingredients'])
# df['directions'] = remove_linebreaks(df_rec['directions'])
# df['categories'] = remove_linebreaks(df_rec['categories'])
# # TODO process desc (contains NaNs)
# df['desc'] = remove_linebreaks(df_rec['desc'])
# df['url'] = remove_linebreaks(df_rec['url'])
# # TODO process servings (is list now)
# #df['servings'] = remove_linebreaks(df_rec['servings'])



# df.to_csv(r'D:\data science\nutrition\data\epi_recipes_with_ghg.csv')



## Okay, reading the data in as json and converting to dataframe gives me
## a much nicer structure for ingredients, categories etc. (lists of text),
## than if I read in the data from csv. 
##
## Given that I can merge the unprocessed data (from json) with the processed
## data using recipe_ID, I can simply replace these columns in the processed 
## data:
	
## Get unprocessed data:
with open(r'D:\data science\nutrition\epi_recipes_detailed', 'r') as io:
        data = json.load(io)
		
# Data structure: 
# Each recipe is an item in a list like this:
#
# [{'title': 'Scallion Pancakes With Chili-Ginger Dipping Sauce ',
#   'ingredients': [
#        '1 (½") piece ginger, peeled,  thinly sliced',
#        '2 Tbsp. low-sodium soy sauce'
# 	               ],
#   'directions': [
#        'Whisk ginger, soy sauce, vinegar
#                 ],
#   'categories': [
#        'Bon Appétit',
#        'Pan-Fry'
#                 ],
#   'date': '2020-05-18T13:50:11.682Z',
#   'desc': 'These pancakes get their light...',
#   'rating': 0.0,
#   'calories': 330.0,
#   'sodium': 462.0,
#   'fat': 17.0,
#   'protein': 5.0}]
df_up = test = pd.DataFrame(data)

## Get processed data
df_p = pd.read_csv(r'D:\data science\nutrition\data\epi_recipes_with_ghg.csv',
						   index_col=False)
df_p.drop(columns=['Unnamed: 0'], inplace=True)

## In SQL the index should be .index, explicitly add it here.
df_p['recipesID'] = df_p.index


## Identify columns to be "replaced" in df_processed from df_unprocessed
coi = ['ingredients', 'directions', 'categories', 'servings', 'desc']


## Drop columns to be replaced from processed dataframe
df_p.drop(columns=coi, inplace=True)


## convert 'desc' to list of lists (so I can delete whitespaces with list_to_text)
df_up['desc'] = [[l] for l in df_up['desc']]


## Create merged dataframe from which to extract the columns of interest
df_merged = df_p.merge(df_up.loc[df_p['index'], coi], 
					   left_on='index', # merge on left 'index' column
					   right_index=True, # and right index
					   how='left')



# Make sure I can add the text columns that contain lists as text in sql
# Write all recipe ingredient text-lines to one long text file. I am not
# super sure how happy sql will be with linebreaks in there, so I will ommit
# line breaks and instead add a semicolon for each new line. 


# Definitely delete linebreaks \n
def remove_linebreaks(text):
	# replace linebreaks
	if text:
		if type(text)==str:
			text = text.replace('\n', ' ').replace('\r', ' ')
		else:
			text = ' '
	return text

# SQL also complains about opened and unclosed quotes ("), which is used as
# inches sometimes
def escape_quotes(text):
	# replace linebreaks
	if text:
		if type(text)==str:
			text = text.replace('"', '\"').replace("'", "\'")
		else:
			text = ' '
	return text

# Convert list of list of strings to list of text (lines separated by ;s)
def list_to_text(LL):
	'''
	Takes a list of lists of strings and returns a list of strings, where the
	individual strings from the input for each list entry have been combined
	with a semicolon (no spaces)
	
	Parameters
	----------
	LL : list of lists
		each element of a sublist is a string
		
	Returns
	-------
	text_list : list or strings (combined sublists with semicolons)
	'''
	failids = []
	text_list = []
	for irec, lines in enumerate(LL):
		
		if irec % 1000 == 0:
			print('Recipe #', irec)
			
		# Exceptions (for empty entries simply move on)
		try:
			if type(lines)==str:
				lines = [lines]
				
			# Remove linebreaks,
			# convert everything to lower case and delete leading or lagging
			# linebreaks or whitespaces
			if len(lines) == 1:
				lines = [remove_linebreaks(lines[0])]
				#lines = [escape_quotes(lines[0])]
				lines = [lines[0].lower().strip()]
				text_list.append(lines[0])
			else:
				lines = [remove_linebreaks(l) for l in lines]
				#lines = [escape_quotes(l) for l in lines]
				lines = [x.lower().strip() for x in lines]
				text_list.append(';'.join(lines)) # combine list elements with ;
			
		except:
			failids.append(irec)
			text_list.append(None)
		
	print('A total of', len(failids), 'entries at', failids,'that did not contain strings.')
		
	return text_list


## remove linebreaks from 'title' column (happens at least once)
df_merged['title'] = [remove_linebreaks(l) for l in df_merged['title']]



## Prepare dataframe for export to SQL database
print(df_merged.columns)
df_sql = df_merged.copy()
for col in coi:
	df_sql[col] = list_to_text(df_merged[col])
	
	
## SQL complains about unquoted line breaks for certain entries, remove them
## OLD: sql_complains = [11390-2, 11415-1, 11501, 11514+1] # -2 converts to python indexing and considers previous line
## NOW: 12606-2
## NOW: 27616-2
sql_complains = []

## Save df_sql to file (and hopefully be able to import it to postgresql db)
## To do so, reorder accordin to sql table schema:
df_sql.iloc[27616-2]['directions']
df_sql.iloc[27616-2]['ingredients']
df_sql.iloc[27616-2]['categories']
df_sql.iloc[27616-2]['servings']
df_sql.iloc[27616-2]['desc']
df_sql.iloc[27616-2]['title']



## No ending quote for line 11368-2
df_sql.iloc[11368-2]['directions']
df_sql.iloc[11368-2]['ingredients']
df_sql.iloc[11368-2]['desc']
df_sql.iloc[11368-2]['title']
df_sql.iloc[11368-2]['servings']







df_sql = df_sql.loc[:, ['recipesID', 'title', 'ingredients', 'directions', 
						'categories', 'date', 'desc', 'rating', 'calories', 
						'sodium', 'fat', 'protein', 'ghg', 'prop_ing', 
						'ghg_log10', 'url', 'servings', 'index']]
df_sql.drop(sql_complains).to_csv(r'D:\data science\nutrition\data\recipes_sql.csv', index=0)






## NOTE: I uploaded df_sql to SQL as recipes now, but I had to still delete a
## a few rows by hand (using Notepad, around 10 in total), because postgres
## threw errors. Probably to do with unclosed quotation marks, but might not
## be worth fixing if I only loose 10 recipes from this... this does mean
## that if I ever want to do this again, I will need to manually go through 
## delete the rows complained about by pgadmin before being able to upload. 






## Verify that df_sql (pretty much my final data to be used) has sensible 
## ghg estimates for each recipe:
from random import randint
def verify_emission_scores(df):
	r = randint(0, len(df))
	print(df.iloc[r]['ingredients'])
	print(' ')
	print('Emission estimate =', df.iloc[r]['ghg'])
	
verify_emission_scores(df_sql)
	

## Load ingredient data
df_ing = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv')

## Load older version of ingredient data
df_ing_old = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')


## Load cleaned recipe data (before adding ghg estimates)
df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_clean.csv',
					 index_col=0)

## Sometimes they don't match because df_ing contains both numbers and strings
## as recipe_ids. There are also still 1/3 etc. in there sometimes...
## For now, just convert to str...
df_ing['recipe_id'] = [str(i) for i in df_ing['recipe_id']]




## Map ingredient labels to each other (recipes to ingredients dfs)
def compare_random_row(df_sql, df_ing):
	r = randint(0, len(df_sql))
	print('-----------------------------')
	print(df_sql.loc[r,'title'])
	print('-----------------------------')
	print('--- Ingredients from df_sql ---')
	print(df_sql.loc[r, 'ingredients'])
	print('Emission estimate =', df_sql.loc[r, 'ghg'])
	print(' ')
	print('--- Ingredients from df_ing ---')
	print(df_ing.loc[df_ing['recipe_id']==str(r),['input', 'ghg_new']])
	print('Emission estimate =', sum(df_ing.loc[df_ing['recipe_id']==str(r),'ghg_new']))


compare_random_row(df_sql, df_ing)



## Loop through all recipe_IDs and check whether ingredients and recipes
## match
def test_emission_scores(df_rec, df_ing):
	'''
	Asserts that the green house gas emissions for a recipe match between the
	recipes and ingredients dataframes for all recipes in df_rec
	
	Parameters
	----------
	df_rec : pandas dataframe
		recipes dataframe 
	df_ing : pandas dataframe
		ingredients dataframe
	'''
	print('Asserting similarity between green house gas emissions of recipes and ingredients dataframes')
	for idx in df_rec.index:
		if idx % 1000 == 0:
			print(idx)
		try:
			A = round(1000*df_rec.loc[idx,'ghg'])
			B = round(1000*np.nansum(df_ing.loc[df_ing['recipe_id']==str(idx),'ghg_new']))
			assert(A==B)
		except AssertionError:
			print('Could not assert emissions to be similar for recipe_id', idx)
			
test_emission_scores(df_sql, df_ing)
	




## Ok they all match... Let's move on to the user reviews






###########
## Prepare user review data for SQL table

## Load user reviews 
reviews = pd.read_csv(r'D:\data science\nutrition\epi_reviews_75plus_w_usernames.csv', index_col=0)
reviews = reviews.loc[:,'user':'rating']


## Add recipeID to reviews dataframe
## To do this properly for SQL I need to import the data from SQL, because I
## deleted a few rows by hand, because of " errors, which are not deleted in 
## df_sql at the moment.
recipes = pd.read_csv(r'D:\data science\nutrition\data\recipes_exported_from_sql.csv')
recipes_sub = recipes.loc[:,['recipesID', 'url']]
recipes_sub['url'] = [r.strip() for r in recipes_sub['url']]


## Merge recipesID into reviews
reviews = reviews.merge(recipes_sub, 
					 how='left', 
					 left_on='title', 
					 right_on='url')


## Remove linebreaks and whitespaces from usernames and make sure they are all strings
usernames = [str(u).strip().replace('\n', '').replace('\r', '') for u in reviews['user']]


## Create final reviews dataframe for SQL (without review text for the moment)
reviews_sql = reviews.loc[:,['user', 'rating', 'recipesID']].copy()
reviews_sql['user'] = usernames

## Remove rows for which I had excluded recipes
reviews_sql['recipesID'] = reviews_sql['recipesID'].astype('int')
reviews_sql.dropna(subset=['recipesID'], inplace=True)

## In this case I am going to use the index as the SQL primary key
reviews_sql.to_csv(r'D:\data science\nutrition\data\reviews_sql.csv', index=1)






## Let's see how I can prepare the data for my web-app:
	
# Load in recipes and reviews data frames
recipes = pd.read_csv(r'D:\data science\nutrition\data\recipes_sql.csv', index_col=0)
reviews = pd.read_csv(r'D:\data science\nutrition\data\reviews_sql.csv', index_col=0)


# Convert recipes dataframe to category dummy coded data frame
# 1.) get categories (columns of dataframe))
sublist = "categories"
column_names = set()
for item in recipes[sublist]:
	cur_elements = str(item).split(';')
	column_names = set(list(set(cur_elements)) + list(column_names))
print('There are', len(column_names), 'recipe categories in the dataset.')


# 2.) fill in dummies (1 when in category otherwise 0)
# pretty inefficient code, but not a problem (less than 1 min still)
dummy_mat = np.zeros((recipes.shape[0],len(column_names)), dtype=np.float32)
for i, item in enumerate(recipes[sublist]):
	
	# progress
	if i % 1000 == 0:
		print('Recipe', i, 'out of', recipes.shape[0])
		
	cur_elements = str(item).split(';')
	for j, name in enumerate(column_names):
		if name in cur_elements:
			dummy_mat[i,j] = 1
			
			
# 3.) Create dummy coded dataframe (not really necessary, but kind of nice to
# have I ever want to look up what a column is)
df_dummy = pd.DataFrame(dummy_mat)
df_dummy.columns = list(column_names)


# 4.) Create similarity matrix using cosine similarity
SM = cosine_similarity(csr_matrix(dummy_mat), dense_output=False)


# takes ~10 min to run (2.4 GB)
scipy.sparse.save_npz('./data/content_similarity.npz', SM)


# 5.)  Keep only the 200 most similar recipes for each recipe
def find_related_recipes(name, df_rec, N, SM):
	rec_id = df_rec.index[df_rec['url'] == name]
	similarities = np.flip(np.sort(SM[rec_id].todense(),axis=1)[:,-N:])
	similarities = np.squeeze(np.asarray(similarities))
	rel_rec_ids = np.flip(np.argsort(SM[rec_id].todense(),axis=1)[:,-N:])
	rel_rec_ids = np.squeeze(np.asarray(rel_rec_ids))
	return(rel_rec_ids, similarities)

N = 200
IDs_pruned = np.zeros((SM.shape[0], N), dtype=np.int16)
SM_pruned = np.zeros((SM.shape[0], N), dtype=np.float32)
for i, name in enumerate(recipes['url']):
	
	if i % 1000 == 0:
		print('Retrieving the', N, 'most similar recipes for recipe', i, name)
		
	IDs_pruned[i,:], SM_pruned[i,:] = find_related_recipes(name, recipes, N, SM)
	
	
# save to file
df_ids = pd.DataFrame(IDs_pruned)
df_sm = pd.DataFrame(SM_pruned)
df_ids.to_csv('./data/content_similarity_200_ids.csv')
df_sm.to_csv('./data/content_similarity_200.csv')




# Test some code to use content based similarity for making predictions
# This will be used in the web-app

# recipe and review dataframes
recipes = pd.read_csv(r'D:/data science/nutrition/data/recipes_sql.csv', index_col=0)
reviews = pd.read_csv(r'D:/data science/nutrition/data/reviews_sql.csv', index_col=0)

# content based similarity
CS_ids = pd.read_csv(r'D:/data science/nutrition/data/content_similarity_200_ids.csv', index_col=0)
CS = pd.read_csv(r'D:/data science/nutrition/data/content_similarity_200.csv', index_col=0)



# Make sure all rows in servings are strings
recipes['servings'].fillna(' ', inplace=True)
# replace double space with double underscore, then replace space with '',
# then replace double underscore with space
# servings = [serv.replace('  ', '__').replace(' ', '').replace('__', ' ') for serv in recipes['servings']]
# recipes['servings'] = servings
recipes['servings'] = [str(serv).replace(';', ' ') for serv in recipes['servings']]

# Make sure date is seen as datetime object
# from dateutil.parser import parse
# datetime = [parse(date) for date in recipes['date']]
recipes.to_csv(r'D:/data science/nutrition/data/recipes_sql.csv')



# Users will search for recipes with key-words, implement search
# For now keep it very simple and assume they know the recipe url


# Make predictions / recommendations based on content based similarity
def content_based_recommendations(url):
	'''Given a recipe url, give me the N most similar recipes'''
	
search_term = recipes['url'][1239]
recipes.index[recipes['url']==search_string].values

Nsim = 200
recipe_id = recipes.index[recipes['url'] == search_term].values
similar_recipe_ids = abs(CS_ids.loc[recipe_id, :]).values
results = recipes.loc[similar_recipe_ids[0], :]
results['similarity'] = CS.loc[recipe_id, :].values[0]


'yemeni-spice-rub-240754' in list(recipes['url'])




## eof
