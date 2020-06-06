# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:07:01 2020

Uses a json file with a list of dictionaries of recipe information from 
www.epicurious.com (created e.g. in scrape_epicurious_recipes.py)

@author: sbuer
"""

# Allow OS manipulations (e.g. deleting files)
import os

# Add project folder to search path
import sys
sys.path.append(r'D:\data science\nutrition\scripts\tdi_challenge_may2020')

# pandas
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 20)

# numpy
import numpy as np

# Data management
import pickle
import json

# Check execution time
import time


# Helper functions to convert json to dummy data frame
from categories_to_dummy import sublists_to_binaries, sublist_uniques



# Load recipe data
with open('epi_recipes_detailed', 'r') as io:
        data = json.load(io)
		
		
		
# First, convert Json-formatted list into a pandas dataframe:
#
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


# I can convert this to a dataframe straight off the bat
df = pd.DataFrame(data)

# Show some basic info of the data
print(df.head())
print(df.columns)
print(df.shape)
print(df.describe())


# As we can see title, rating, cal, sod, fat and protein, as well as date and
# description already look like sensible single value per row columns.
# The columns ingredients, directions and categories are currently lists with
# possibly multiple values per cell and might need a bit more work.


# Using the functions provided from
# https://www.kaggle.com/hugodarwood/epirecipes?select=recipe.py
# convert a sublist to dummy, i.e. each item becomes a new column in the df
# and recipes (rows) that have the item are denoted with 1, otherwise 0.


# Test run:
sublist = "categories"
df_dummy = sublists_to_binaries(df.iloc[0:100,:],sublist)

print([x for x in df_dummy.columns])



# --- Takes around 45 min to run ---

# Seems to be working really well. Do this for all data:
# df_dummy = sublists_to_binaries(df,sublist) 

# Save dataframe to csv
# df_dummy.to_csv('epi_recipes')

# ----------------------------------




# It would be great if instead of dummy coded category values (which includes
# ingredients) I could pinpoint the exact amount of each ingredient.
# I should be able to leverage the ingredients column for this. 



# Create look-up table for ingredient amounts:
	
# Actually, I found a tool that can maybe do most if not all of this for me.
# See: https://hub.docker.com/r/mtlynch/ingredient-phrase-tagger/
# And: https://mtlynch.io/resurrecting-1/
# And the original codebase from NYT: 
# https://github.com/NYTimes/ingredient-phrase-tagger

# It seems somewhat cumbersome, because it uses an ubuntu docker container
# that then uses python 2.7 code in the comman line together with crf++, which
# is a conditional random field toolbox written in C++ that needs to be trained
# on the NYT database (or a different database, but this one is conveniently
# included). 

# Building the model works pretty straightforwardly in the docker cli by 
# running the following from the project directory:
	
# docker pull mtlynch/ingredient-phrase-tagger
# docker run -it mtlynch/ingredient-phrase-tagger bash

# # Train a new model
# MODEL_DIR=$(mktemp -d)
# bin/train-prod-model "$MODEL_DIR"
# MODEL_FILE=$(find $MODEL_DIR -name '*.crfmodel')

# # Parse some ingredients
# echo '
# 2 tablespoons honey
# 1/2 cup flour
# Black pepper, to taste' | bin/parse-ingredients.py --model-file $MODEL_FILE



## Create a test .txt file with ingredient instructions to pipe into the model
 
# Get ingredients of first recipe in list
lines = df['ingredients'][0]

# Convert everything to lower case
lines = [x.lower() for x in lines]

# Replace ½ with 1/2 etc. to conform to utf8
def replace_non_utf8(text):
	non_utf8 = dict()
	non_utf8['½'] = ' 1/2'
	non_utf8['¼'] = ' 1/4'
	non_utf8['¾'] = ' 3/4'
	non_utf8['⅔'] = ' 2/3'
	non_utf8['⅓'] = ' 1/3'
	for i, char in enumerate(text):
		if char in non_utf8.keys():
			text = text.replace(char, non_utf8[char]).strip()
	return text

# Definitely delete linebreaks \n
def remove_linebreaks(text):
	return text.replace('\n', ' ')

# Also make sure units can be understood
def standardize_units(text):
	units = {"cup": ["cups", "c.", "c"], 
			 "fluid_ounce": ["fl. oz.", "fl oz", "fluid ounces"],
	         "gallon": ["gal", "gal.", "gallons"], 
			 "ounce": ["oz", "oz.", "ounces"],
	         "pint": ["pt", "pt.", "pints"], 
			 "pound": ["lb", "lb.", "pounds"],
	         "quart": ["qt", "qt.", "qts", "qts.", "quarts"],
	         "tablespoon": ["tbsp.", "tbsp", "T", "T.", "tablespoons", "tbs.", "tbs"],
	         "teaspoon": ["tsp.", "tsp", "t", "t.", "teaspoons"],
	         "gram": ["g", "g.", "gr", "gr.", "grams"], 
			 "kilogram": ["kg", "kg.", "kilograms"],
	         "liter": ["l", "l.", "liter", "liters"], 
			 "milligram": ["mg", "mg.", "milligrams"],
	         "milliliter": ["ml", "ml.", "milliliters"], 
			 "pinch": ["pinches"],
	         "dash": ["dashes"], 
			 "touch": ["touches"], 
			 "handful": ["handfuls"],
	         "stick": ["sticks"], 
			 "clove": ["cloves"], 
			 "can": ["cans"], 
			 "scoop": ["scoops"], 
			 "filets": ["filets"], 
			 "sprig": ["sprigs"],
			 "slice": ["slices"],
			 "stalk": ["stalks"],
			 "piece": ["pieces"]}
	# go through keys (cup, fuild_ounces)
	for key in list(units.keys()):
		# go through key values (cups, cup, c., ...)
		for syn in units[key]:
			# replace with standardized unit values (cup)
			if ' '+syn+' ' in text:
				text = text.replace(syn, key)
	return text

# Apply utf8 correction and unit standardization
lines = [replace_non_utf8(l) for l in lines]
lines = [standardize_units(l) for l in lines]	
lines
			


# Write to file. 
f=open('recipe1_ingredients.txt', 'w')
f.writelines("%s\n" % replace_non_utf8(i) for i in lines)
f.close()


# I can then export this to the docker container using the following:
# Open a new docker terminal

# $docker ps # shows currently open containers - use name for mycontainer
# $docker cp ../../recipe1_ingredients.txt mycontainer:/recipe1_ingredients.txt

# I can read this in in my docker environment and save it to file using this:
#:/app# cat ../recipe1_ingredients.txt | bin/parse-ingredients.py --model-file $MODEL_FILE > crf_output.txt

# Copy back to Windows from Docker container:
# $docker cp mycontainer:/app/crf_output.txt ../../crf_output.txt




# Remove recipes without ingredients
# Replace empty ingredient lists with NaN, then drop NaNs
df.dropna(subset=['ingredients'], inplace=True)
df['ingredients'] = df['ingredients'].apply(lambda y: np.nan if len(y)==0 else y)
df.dropna(subset=['ingredients'], inplace=True)
		

# Write all recipe ingredient text-lines to one long text file
def write_ingredients(outfile, ingredients):
	'''
	Parameters
	----------
	outfile : string
		path and filename (e.g. examplefile.txt) to write to.
	ingredients : pandas.core.series.Series
		'ingredients' column of data frame containing a list of ingredients
		for each recipe (i.e. data frame row)

	Returns
	-------
	rec_id : list
		list of recipe IDs associated with each ingredient line of text.

	'''
	# Delete file if it already exists (to avoid appending to existing file)
	os.remove(outfile)
	rec_id = list()
	for irec, lines in enumerate(ingredients):
		
		if irec % 1000 == 0:
			print('Recipe #', irec)
			
		# Convert everything to lower case
		lines = [x.lower() for x in lines]
		
		# Remove linebreaks
		lines = [remove_linebreaks(l) for l in lines]
			
		# Apply utf8 correction and unit standardization
		lines = [replace_non_utf8(l) for l in lines]
		lines = [standardize_units(l) for l in lines]	
		
		# Keep Recipe ID vector to later associate lines with recipes
		rec_id += list(np.tile(irec,len(lines)))
		
		# Write to file. 
		f=open(outfile, 'a', encoding='utf-8')
		f.writelines("%s\n" % l for l in lines)
	f.close()
		
	return rec_id



# Write recipe ingredients to file
rec_id = write_ingredients('recipe_ingredients.txt', df['ingredients'])
	
# The process is killed after 5-10 minutes in the unix docker environment
# when I put in all data at once into crf++ so let's try putting in half:
rec_id1 = write_ingredients('recipe_ingredients_part1.txt', df['ingredients'][0:18000])
rec_id2 = write_ingredients('recipe_ingredients_part2.txt', df['ingredients'][18000:])
	



# Check if recipe ID and number of lines written to file are consistent
def textfile_count(filename):
	'''
	Print out number of lines, words and characters in .txt file

	'''
	file = open(filename, "r", encoding='utf-8')
	
	number_of_lines = 0
	number_of_words = 0
	number_of_characters = 0
	for line in file:
	  line = line.strip("\n")
	
	  words = line.split()
	  number_of_lines += 1
	  number_of_words += len(words)
	  number_of_characters += len(line)
	
	file.close()
	
	print("lines:", number_of_lines, 
		  "words:", number_of_words, 
		  "characters:", number_of_characters)



# Check rec_id length versus line count
textfile_count('recipe_ingredients.txt')
print('Number of recipe IDs: ', len(rec_id))

# Check part1
textfile_count('recipe_ingredients_part1.txt')
print('Number of recipe IDs: ', len(rec_id1))

# Check part2
textfile_count('recipe_ingredients_part2.txt')
print('Number of recipe IDs: ', len(rec_id2))




## Read in CRF output:
df1 = pd.read_json('crf_output_part1.txt')
df2 = pd.read_json('crf_output_part2.txt')


## They do not match - in fact there is a mismatch of 1 less in the crf output
## from the first part (< 18000)
## So let's go through the output and check for each line if it matches with
## the input:
from Levenshtein import distance
	
f = open("recipe_ingredients_part1.txt", "r", encoding='utf-8')
text_input = f.readlines()
crf_output = df1['input']
equal_count = 0
unequal_ids = list()
for irec, (textin, textout) in enumerate(zip(text_input, crf_output)):
	
	if irec % 1000 == 0:
		print('Recipe #', irec)

	edit_dist = distance(textout, textin.replace('\n', ''))
	min_len = np.min([len(textout), len(textin.replace('\n', ''))])
	
	if min_len == 0:
		print(irec, "Output from CRF: ", textout)
		print(irec, "Input text: ", textin)
		
	if (min_len-edit_dist)/min_len > 0.5:
		equal_count += 1
	else:
		unequal_ids.append(irec)
	
	
## Clearly, item 110477 is the culprit: 
# 110477 Output from CRF fresh tarragon and mint sprig (for garnish)
# 110477 Input text 
## Delete entry and move on
if len(rec_id) == 354202:
	del rec_id[110477] 


## Recombine df1 and df2, then finally add recipe ID
df_crf = pd.concat([df1, df2], axis= 0)
df_crf['recipe_id'] = rec_id


## Check that I didn't break anything
df_crf['input'][df_crf['recipe_id'] == rec_id[110477]]
df['ingredients'].iloc[rec_id[110477]]


## Reset index
df.reset_index(inplace=True)


## Save recipe df (difference to input is that recipes with missing ingredients 
## were dropped)
df.to_csv('epi_recipes_clean.csv')


## Save ingredients df (can be mapped to df - could also be mapped to the input
## df using .iloc)
df_crf.to_csv('crf_ingredients_table.csv')




##############################################################################

## Reload data (note that by saving and loading I destroy the nice list
## structure of the categories column)
df = pd.read_csv('epi_recipes_clean.csv')
df_crf = pd.read_csv('crf_ingredients_table.csv')


# Now that I have some estimate of the quantities of ingredients in my recipes
# I would ideally be able to look up the quantities for each ingredient for 
# each recipe. However, the ingredient "categories" are currently still based 
# on the "categories" column, which may not exactly correspond to the 
# ingredients from the "ingredients" column. Let's explore that first. 

df_crf.head()


ingredient_names = df_crf['name'].unique()
print("I have", len(ingredient_names), "unique ingredient labels.")


# remove nans
ingredient_names = [x for x in ingredient_names if str(x) != 'nan']
print("I have", len(ingredient_names), "unique ingredient labels after removing nans.")



# Just under 30000 ingredient labels may be a little bit much to handle, can
# I trim this down by summarizing some groups?

# To start off, I can substitue longer strings like
# "juice of 1 medium lime"
# with
# "lime"
# This can be done by checking a standard ingredient corpus and seeing if one
# of the ingredients is part of the string ("lime" is part of "juice of 1 medium
# lime"). However, when two or more matches are found, I should not interfere.


# Get corpus of ingredient categories
corpus = []
for categories in df['categories']:
	[corpus.append(item) for item in categories if item not in corpus]
	

# Go through ingredient labels and attach each associated category label
ingredient_dict = dict()
for ingredient in ingredient_names:
	for category in corpus:
		if category.lower() in ingredient:
			if ingredient in ingredient_dict
				ingredient_dict[ingredient].append(category)
			else:
				ingredient_dict[ingredient] = [category]
			
			
# Prune terms that are subterms of others, e.g. we want to keep Olive oil 
# instead of Olive, Hazelnut instead of nut!



# Be careful with situations like Black Pepper vs Pepper, where only Pepper is
# in the categories list!


# Ingredients with exactly 1 category label can be substituted for that label




# Adapted code from
# https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups
# and
# https://science.sciencemag.org/content/315/5814/972
from sklearn.cluster import AffinityPropagation
from distance import levenshtein

words = ingredient_names[0:100]
#words = np.asarray(words) #So that indexing with a list will work
lev_sim = -1*np.array([[levenshtein(w1,w2) for w1 in words] for w2 in words])

aff_prop = AffinityPropagation(affinity="precomputed", damping=0.5)
aff_prop.fit(lev_sim)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[aff_prop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


















# eof

