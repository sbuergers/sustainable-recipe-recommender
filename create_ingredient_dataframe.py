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
pd.set_option('display.max_rows', 50)
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
# some ingredients) I could pinpoint the exact amount of each ingredient.
# I should be able to leverage the ingredients column for this. 



# Create look-up table for ingredient amounts:
	
# Actually, I found a tool that can maybe do most if not all of this for me.
# See: https://hub.docker.com/r/mtlynch/ingredient-phrase-tagger/
# And: https://mtlynch.io/resurrecting-1/
# And the original codebase from NYT: 
# https://github.com/NYTimes/ingredient-phrase-tagger

# It seems somewhat cumbersome, because it uses an ubuntu docker container
# that then uses python 2.7 code in the command line together with crf++, which
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



## I found an ingredient database from the agricultural research service:
## https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/methods-and-application-of-food-composition-laboratory/mafcl-site-pages/sr11-sr28/ 
## It has just over 8000 unique ingredient labels. Try to match my labels
## to those standard ones:
istd = pd.read_csv('ingredient_labels_standard.csv')

## First attempt, simply convert everything to lower and assign label that
## most closely matches a standard label:
standard_labels = list(istd.Shrt_Desc)
standard_labels = [x.lower() for x in standard_labels]

## The standard labels can be quite long, but the main ingredient name is 
## before the first comma, and maybe before the second, e.g. 
## oil, avocado, ... ; so let's get all unique before first comma labels and
## before second comma labels, i.e. both oil and oil, avocado

## It is also important to note that the best fit might be either first and
## then second term or second and then first term, like with oil, avocado. 

## Create a standard ingredient label list including only the first and second
## term from standard_labels (before first comma and before second comma).
## Also add labels that are flipped (oil, olive becomes olive oil)
pruned_label = []
for label in standard_labels:
	L = label.split(',')
	if len(L) > 1:
		pruned_label.append(L[0] + ' ' + L[1])
		pruned_label.append(L[1] + ' ' + L[0])
	pruned_label.append(L[0])

## Drop duplicates
pruned_label = list(np.unique(pruned_label))

print('There are', len(pruned_label), 'standard ingredient labels to search')


# Compute edit distance between crf labels and standard labels
from Levenshtein import distance 

def edit_dist(L1, L2, verbose=False):
	'''
	Parameters
	----------
	L1, L2 : List
		List of strings.
	verbose : Boolean
		If True prints out progress
	Returns
	-------
	D : Numpy array
		Edit distance matrix between L1 and L2
	'''
	D = np.zeros((len(L1), len(L2)))
	for i in range(0, len(L1)):
		if verbose:
			if i % 100 == 0:
				print('Computing pairwise edit distance for row', i, 'out of', len(L1))
		for j in range(0, len(L2)):
			D[i,j] = distance(L1[i], L2[j])
	return D

# Compuate pairwise edit distances
D = edit_dist(ingredient_names[0:100], pruned_label, verbose=True)



# Check if we get sensible output
for i, item in enumerate(ingredient_names[0:100]):
	best_match = pruned_label[int(np.where(D[i,:] == np.amin(D[i,:]))[0][0])]
	print(item, '-', best_match)




## That does not work very well... too many missing standard labels





## Let's just work with the labels I got from my own data:
	
# Remove leading or laggin spaces
corpus = list(map(lambda x: str(x).strip(), ingredient_names))

# Only allow letters and dashes, including accented letters
import re
regex = re.compile("[()-a-zÀ-ÿ0-9a-zA-Z'& ]+")
corpus = [regex.search(x.lower()).group() for x in corpus]
corpus = list(np.unique(corpus))

# Drop NaNs
corpus = [x for x in corpus if str(x) != 'nan']

# Drop stars (*)
corpus = [x.replace('*', '') for x in corpus]

# Drop long sequences (>= 3 words)
corpus = list(filter(lambda a: len(a.split()) < 3, corpus))

# Drop empty elements
corpus = list(filter(lambda x: x != "", corpus))

# Drop elements of length < 3
corpus = list(filter(lambda x: len(x) > 2, corpus))

# Remove commas and periods
corpus = [x.replace('.', '') for x in corpus]
corpus = [x.replace(',', '') for x in corpus]

# Replace dashes with space
corpus = [x.replace('-', ' ') for x in corpus]

# Drop duplicates
corpus = np.unique(corpus)

# # Print corpus to text file and use word's autocorrect
# with open('ingredient_labels_raw.txt', 'w') as f:
#     for item in corpus:
#         f.write('%s\n' % item)



# Anything we can collapse together? Check pairwise edit distances and inspect
# terms with edist of 1 to see whether they should be the same
# Note: distance from Levenstein is 28 times faster than levensthein from the 
# distance module
from Levenshtein import distance 

def pairwise_edit_dist(L, verbose=False):
	'''
	Parameters
	----------
	L : List 
		List of strings.
	verbose : Boolean
		If True prints out progress
	Returns
	-------
	D : Numpy array
		Pairwise edit distance upper triangular matrix.
	'''
	D = np.zeros((len(L), len(L)))
	for i in range(0, len(L)):
		if verbose:
			if i % 100 == 0:
				print('Computing pairwise edit distance for row', i, 'out of', len(L))
		for j in range(i, len(L)):
			D[i,j] = distance(L[i], L[j])
	return D

# Compuate pairwise edit distances
D = pairwise_edit_dist(corpus, verbose=True)


# Show pairs with edit distances of 1
print("Total number of pairs with edit distance of 1:", np.sum(D == 1))
pairs = np.where(D == 1)
for i, j in zip(pairs[0], pairs[1]):
	print(corpus[i], ' - ', corpus[j])
			
# Only keep the first entry and drop others (there will be a few errors here,
# especially for very short words, like twine and wine). 
for j in zip(pairs[1]):
	corpus[j[0]] = ""

# Remove duplicates
corpus = list(filter(lambda x: x != "", corpus))	

	

# Go through ingredient labels and attach each associated category label
ingredient_dict = dict()
for i, ingredient in enumerate(ingredient_names):
	if i % 100 == 0:
		print('Ingredient #', i)
	for category in corpus:
		if ' '+category.lower()+' ' in ' '+ingredient+' ':
			if ingredient in ingredient_dict:
				ingredient_dict[ingredient].append(category)
			else:
				ingredient_dict[ingredient] = [category]
			
			
# Prune terms that are subterms of others, e.g. we want to keep Olive oil 
# instead of Olive, Hazelnut instead of nut!
for item in ingredient_dict:
    # create copy of current list and sort by item length (descending)
	values = sorted(ingredient_dict[item].copy(), key=len, reverse=True)
	while len(values) > 1:
		# Compare the shortest string to all longer strings
		val = values.pop()
		for longer_val in values:
			if val.lower() in longer_val.lower():
				if val.lower() in ingredient_dict[item]:
					ingredient_dict[item].remove(val)
				
				
## Remove measurements (numbers and units - '100' or 'milliliters')
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

# count how many items have multiple matches
multiple_matches = 0
for item in ingredient_dict:
	if len(ingredient_dict[item]) > 1:
		multiple_matches += 1
	for val in ingredient_dict[item]:
		if val in list(units.keys()):
			ingredient_dict[item].remove(val) # this does not work properly, not sure why
		if val.isdigit():
			ingredient_dict[item].remove(val)
print('How many items have multiple matches? -->', multiple_matches)


# Simply select the longest string for each ingredient (of course not ideal,
# but should work in most cases)
for item in ingredient_dict:
	if len(ingredient_dict[item]) > 1:
		ingredient_dict[item] = [sorted(ingredient_dict[item].copy(), 
									  key=len, reverse=True)[0]]
				

## How many unique labels do I have now?
ingredients = [item for L in ingredient_dict.values() for item in L]
ingredients = list(np.unique(ingredients))
len(ingredients) 


## Save to file. I can go through it manually now and assign more concise
## labels
with open('ingredient_labels_raw.txt', 'w') as f:
    for item in ingredients:
        f.write('%s\n' % item)



## Assign new labels
## Go through original labels from crf and assign closest match in ingredients

# Compute edit distance between crf labels and processed labels
from Levenshtein import distance 

def edit_dist(L1, L2, verbose=False):
	'''
	Parameters
	----------
	L1, L2 : List
		List of strings.
	verbose : Boolean
		If True prints out progress
	Returns
	-------
	D : Numpy array
		Edit distance matrix between L1 and L2
	'''
	D = np.zeros((len(L1), len(L2)))
	for i in range(0, len(L1)):
		if verbose:
			if i % 100 == 0:
				print('Computing pairwise edit distance for row', i, 'out of', len(L1))
		for j in range(0, len(L2)):
			D[i,j] = distance(L1[i], L2[j])
	return D

# Compuate pairwise edit distances
D = edit_dist(ingredient_names[0:100], ingredients, verbose=True)

# Check if we get sensible output
for i, item in enumerate(ingredient_names[0:100]):
	best_match = ingredients[int(np.where(D[i,:] == np.amin(D[i,:]))[0][0])]
	print(item, '-', best_match)
	
	
	
# Looks to match pretty well! Do it for all ingredients.
D = edit_dist(ingredient_names, ingredients, verbose=True)

# Collect best matches
match = list()
for i, item in enumerate(ingredient_names):
	match.append(ingredients[int(np.where(D[i,:] == np.amin(D[i,:]))[0][0])])

# Now assign these labels to the ingredient items in df_crf
df_map = pd.DataFrame({'name':ingredient_names, 'name_pruned':match})
df_pruned = pd.merge(df_crf, df_map, on='name', how='left')	
				


## So now I have the automatically pruned labels included. Let's include the
## manually pruned labels and see how many we got left:
	
## Reload (after adding labels by hand):
df_map2 = pd.read_csv(r'D:\data science\nutrition\ingredient_labels_raw.csv', 
						  names=['name_pruned', 'name_man_pruned1'])			
			
## Add manually pruned labels to df_crf
df_pruned2 = pd.merge(df_pruned, df_map2, on='name_pruned', how='left')

## Get unique manually assigned labels (they are not very good for now, I should
## go through this again and do this more carefully.)
man_labels = df_pruned2['name_man_pruned1'].unique()
print('There are', len(man_labels), 'manually assigned labels after the firsts pass')


## Now I need to go through this manually a second time
## Save labels to new file
with open(r'D:\data science\nutrition\ingredient_labels_scnd_pass.txt', 'w') as f:
    for item in man_labels:
        f.write('%s\n' % item)









# Now assign these labels to the ingredient items in df_pruned2
df_map3 = pd.read_csv(r'D:\data science\nutrition\ingredient_labels_scnd_pass.csv', 
					  names=['name_man_pruned1', 'name_man_pruned2'])
df_pruned3 = pd.merge(df_pruned2, df_map3, on='name_man_pruned1', how='left')	

## Save updated df_crf
df_pruned3.to_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')







## Some unit labels are messed up, fix them manually, too:
unit_labels = df_pruned3['unit'].unique()
with open(r'D:\data science\nutrition\unit_labels.txt', 'w') as f:
    for item in unit_labels:
        f.write('%s\n' % item)
		
		
# Add manually fixed unit labels to dataframe df_pruned3
units = pd.read_csv(r'D:\data science\nutrition\unit_labels.csv',
					names=['unit', 'unit_man'])

df_pruned3 = pd.merge(df_pruned3, units, on='unit', how='left')

## Save updated df_crf
df_pruned3.to_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')





## Write functions to be able to extract GHG emissions of a recipe based on
## ingredients and their quantities 
ghg = pd.read_csv(r'D:\data science\nutrition\GHG-emissions-by-life-cycle-stage.csv')



## Create a look-up table for ingredient amounts
## From wikipedia: https://en.wikipedia.org/wiki/Cooking_weights_and_measures
## Liquid units in milliliters
units_ml = {'drop':0.051,
				'smidgen':0.116,
				'pinch':0.231,
				'dash':0.462,
				'saltspoon':0.924,
				'coffeespoon':1.848,
				'fluid dram':3.697,
				'teaspoon':4.93,
				'dessertspoon':9.86,
				'tablespoon':14.79,
				'ounce':29.57,
				'wineglass':59.15,
				'teacup':118.29,
				'cup':236.59,
				'pint':473.18,
				'quart':946.35,
				'pottle':1892.71,
				'gallon':3785.41,
				'piece':118.29, # this is extremely variable (e.g. ginger, fish, pig, ...)
				'clove':4.93, # A clove of garlic should be around 1 tsp
				'envelope':1.25*29.57, # for yeast
				'pound':16*29.57, # 16 ounces
				'bunch':1.5*29.57, # 1-2 ounces
				'gram':1, # approximately correct, depending on material
				'package':19*29.57, # package of tofu... there is great variability though
				'head':500, # at least a pound for cabbage, cauliflower etc.
				'slice':0.8*29.57, # for a slice of cheese
				'sprig':1.848, 
				'can':12*29.57, # for a small can
				'stick':4*29.57, # for a butter stick
				'strip':29.57, 
				'stalk':59.15, # for a stalk of celery (could also be lemongrass)
				'cube':4.93, # should be 1 teaspoon (always sugar)
				'fillet':100, # 100 grams is roughly 1 fillet
				'handful':118.29, # by definition it's half a cup
				'fistfull':59.15, # by definition it's half a handful
				'bag':11*29.57, # 10-12 ounces it seems, though probably not always
				'loaf':6*236.59, # approximately 4-8 cups
				'bulb':8*29.57, # for a bulb of fennel
				'bottle':1.25*473.18, # could be beer or wine
				'ear':(3/4)*236.59, # for an ear of corn
				'ball':236.59, # a ball of mozzarella is a cup
				'batch':473.18, # pretty unclear, milk, eggs, fish, fruit...
				'sheet':14*29.57, # sheet of pastry dough or cheese
				'dozen':473.18, # used for clams - roughly one pound
				'liter':1000,
				'box':473.18, # e.g. box of chocolate or milk
				'packet':59.15, # can be many sizes, usually quite small (1/4 ounce), but can also be a packet of dumpling wrappers
				'chunk':1.1*14.79, # used for ginger pretty much
				'rack':3.5*16*29.57, # a rack of ribs... 3-4 pounds
				'jar':236.59, # pretty variable...
				'stem':14.79, # a stem of thyme
				'part':118.29, # bread, fruit and sugar... no clear amount, few entries though
				'branch':14.79,
				'inch':1.1*14.79, # used only for ginger
				'wedge':6*29.57, # a wedge of cheese is 4-8 ounces
				'link':29.57, # a link of sausage is 1 ounce
				'square':29.57, # a square of chocolate - 1 ounce
				'knob':118.29, # same as piece
				'scoop':59.15, # as much as a wineglass by definition
				'fifth':757, # a fifth of a gallon for liquor - wow
				'twist':29.57, # refers to lemon or orange peel
				'pair':172*2 # because in this case it's chicken breast !!!
				}

## Some quantity labels are messed up, fix them manually, too:
qty_labels = df_pruned3['qty'].unique()
qty_labels[qty_labels=='1⁄3'] = '1/3'
qty_labels[qty_labels=='1⁄2'] = '1/2'
qty_labels[qty_labels=='1⁄4'] = '1/4'
qty_labels[qty_labels=='1⁄8'] = '1/8'
qty_labels[qty_labels=='1\u20091/2'] = '1 1/2'
qty_labels[qty_labels=='2\u20091/2'] = '2 1/2'
qty_labels[qty_labels=='1‟'] = '1'

with open(r'D:\data science\nutrition\qty_labels.txt', 'w') as f:
    for item in qty_labels:
        f.write('%s\n' % item)
		
		

# Add manually fixed unit labels to dataframe df_pruned3
qty = pd.read_csv(r'D:\data science\nutrition\qty_labels.csv',
					names=['qty', 'qty_man'])

df_pruned3 = pd.merge(df_pruned3, qty, on='qty', how='left')

## Save updated df_crf
df_pruned3.to_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')





## Try to estimate GHG emissions based on ingredients and quantities:

def get_ingredient_emissions():
	
	# retrieve ingredient quantity
	try:
		q = float(df_pruned3.iloc[idx]['qty_man'])
	except:
		print('Could not convert quantity string to number for:', df_pruned3.iloc[idx]['input'])
		q = 0 # can I guess this instead?
		
	# convert quantity to ml (which is also roughly grams)
	qtotal = q * units_ml[df_pruned3.iloc[idx]['unit_man']]
		
	









## Load in manual mappings. Try to find ingredient compounds of dishes that
## are also ingredients (e.g. salsa verde). 
recipe_names = np.asarray([x.lower().strip() for x in list(df['title'])], dtype=object)

## Find exact match
df['title_norm'] = recipe_names
recipe_indeces = df[df['title_norm'] == 'salsa verde'].index

## Look up ingredients of recipes
lbl = df_pruned3[df_pruned3['recipe_id'] == recipe_indeces[0]]['name_man_pruned2']
qty = df_pruned3[df_pruned3['recipe_id'] == recipe_indeces[0]]['qty']
unit = df_pruned3[df_pruned3['recipe_id'] == recipe_indeces[0]]['unit']




## Return recipe names that contain ingredient name
found_in = np.array(['salsa verde' in name for name in recipe_names])
recipe_names[found_in]














# Get corpus of ingredient categories
corpus = []
for categories in df['categories']:
	[corpus.append(item) for item in categories if item not in corpus]
	
	
# The categories are not tailored to reflect all ingredients, try to use a
# true ingredient list, using the NY-times dataset. As this data has similar
# problems to mine, however, only keep ingredients with less than 3 words.
df_nyt = pd.read_csv('nyt-ingredients-snapshot-2015.csv', encoding='utf-8')
corpus = df_nyt['name'].unique()

# Remove leading or laggin spaces
corpus = list(map(lambda x: str(x).strip(), corpus))

# Only allow letters and dashes, including accented letters
import re
regex = re.compile("[()-a-zÀ-ÿ0-9a-zA-Z'& ]+")
corpus = [regex.search(x.lower()).group() for x in corpus]
corpus = np.unique(corpus)

# Drop NaNs
corpus = [x for x in corpus if str(x) != 'nan']

# Drop long sequences (>= 3 words)
corpus = list(filter(lambda a: len(a.split()) < 3, corpus))

# Drop empty elements
corpus = list(filter(lambda x: x != "", corpus))

# Drop elements of length < 3
corpus = list(filter(lambda x: len(x) > 2, corpus))

# Remove commas and periods
corpus = [x.replace('.', '') for x in corpus]
corpus = [x.replace(',', '') for x in corpus]

# Replace dashes with space
corpus = [x.replace('-', ' ') for x in corpus]

# Drop duplicates
corpus = np.unique(corpus)

# Print corpus to text file and use word's autocorrect
with open('ingredient_labels_raw.txt', 'w') as f:
    for item in corpus:
        f.write('%s\n' % item)



# Anything we can collapse together? Check pairwise edit distances and inspect
# terms with edist of 1 to see whether they should be the same
# Note: distance from Levenstein is 28 times faster than levensthein from the 
# distance module
from Levenshtein import distance 

def pairwise_edit_dist(L, verbose=False):
	'''
	Parameters
	----------
	L : List
		List of strings.
	verbose : Boolean
		If True prints out progress
	Returns
	-------
	D : Numpy array
		Pairwise edit distance upper triangular matrix.
	'''
	D = np.zeros((len(L), len(L)))
	for i in range(0, len(L)):
		if verbose:
			if i % 100 == 0:
				print('Computing pairwise edit distance for row', i, 'out of', len(L))
		for j in range(i, len(L)):
			D[i,j] = distance(L[i], L[j])
	return D

# Compuate pairwise edit distances
D = pairwise_edit_dist(corpus, verbose=True)


# Show pairs with edit distances of 1
print("Total number of pairs with edit distance of 1:", np.sum(D == 1))
pairs = np.where(D == 1)
for i, j in zip(pairs[0], pairs[1]):
	print(corpus[i], ' - ', corpus[j])
			
# Only keep the first entry and drop others (there will be a few errors here,
# especially for very short words, like twine and wine). 
for j in zip(pairs[1]):
	corpus[j] = ""

# Remove duplicates
corpus = list(filter(lambda x: x != "", corpus))	

# 
print("Total number of pairs with edit distance of 2:", np.sum(D == 2))
pairs = np.where(D == 2)
for i, j in zip(pairs[0], pairs[1]):
	print(corpus[i], ' - ', corpus[j])	
	

# Go through ingredient labels and attach each associated category label
ingredient_dict = dict()
for i, ingredient in enumerate(ingredient_names):
	if i % 100 == 0:
		print('Ingredient #', i)
	for category in corpus:
		if ' '+category.lower()+' ' in ingredient:
			if ingredient in ingredient_dict:
				ingredient_dict[ingredient].append(category)
			else:
				ingredient_dict[ingredient] = [category]
			
			
# Prune terms that are subterms of others, e.g. we want to keep Olive oil 
# instead of Olive, Hazelnut instead of nut!
for item in ingredient_dict:
    # create copy of current list and sort by item length (descending)
	values = sorted(ingredient_dict[item], key=len, reverse=True)
	while len(values) > 1:
		# Compare the shortest string to all longer strings
		val = values.pop()
		for longer_val in values:
			if val.lower() in longer_val.lower():
				ingredient_dict[item].remove(val)
				
				
# Replace


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
words = np.asarray(words) #So that indexing with a list will work
lev_sim = -1*np.array([[levenshtein(w1,w2) for w1 in words] for w2 in words])

aff_prop = AffinityPropagation(affinity="precomputed", damping=0.5)
aff_prop.fit(lev_sim)
for cluster_id in np.unique(aff_prop.labels_):
    exemplar = words[aff_prop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(aff_prop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


















# eof

