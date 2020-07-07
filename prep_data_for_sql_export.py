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
def verify_emission_scores(df):
	## ...
	
	
	
	
	
	
	
	
###########
## Prepare user data for SQL table



















## Save only columns where line-breaks really shouldn't be a problem for now
df_basic = df.loc[:,['recipesID', 'title', 'date', 'rating', 'calories', 'sodium', 
				   'fat', 'protein', 'ghg', 'prop_ing', 
				   'ghg_log10', 'url', 'index']]
df_basic.to_csv(r'D:\data science\nutrition\data\recipes_sql.csv', index=0)





## Prepare ingredients dataframe

# Load preprocessed ingredient data (from impute_ingredient_ghg_values.py)
df_ing = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv',
					 index_col=1)
print(df_ing.columns)


## Go back to a point where I didn't somehow substitue whole rows with non-
## sense like 1/3
df_ing = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed_firstsave.csv')
	
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
	if i not in df_rec.index:
		ing_to_rec.append(i)
		
for i in df_rec.index:
	if i not in ing_recipe_ids:
		rec_to_ing.append(i)
		
print('Recipe IDs in ingredients table not found in recipes table:', ing_to_rec)
print('Recipe IDs in recipes table not found in ingredients table:', rec_to_ing)


np.setdiff1d(list(ing_recipe_ids), list(df_rec.index.values))






## At this point, do recipe_ids in df_pruned3 (ingredients) and df (recipes)
## match?
recipe_ids_ingredients = sorted([int(l) for l in list(set(df_ing['recipe_id']))])
recipe_ids_recipes = sorted(list(set(df_rec.index)))

assert(len(recipe_ids_ingredients) == len(recipe_ids_recipes))
assert(sorted(recipe_ids_ingredients) == sorted(recipe_ids_recipes))





# Keep only columns for 
# ingredientsID is implied by the index (saved automatically)
# recipe_id = recipeID
# ghg = emissions
# ghg_missing = emissions_missing
# ghg_new = emissions_imputed
df_ing.rename(columns={"Unnamed: 0": "ingredientsID"}, inplace=True)
df_ing_basic = df_ing.loc[:,['ingredientsID', 'recipe_id', 'ghg', 'ghg_missing', 'ghg_new']]
df_ing_basic['ghg_missing'] = df_ing_basic['ghg_missing'].set_type('boolean')
df_ing_basic.to_csv(r'D:\data science\nutrition\data\ingredients_sql.csv', index=False)













