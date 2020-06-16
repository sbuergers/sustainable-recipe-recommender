# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:17:50 2020

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




ghg = pd.read_csv(r'D:\data science\nutrition\GHG-emissions-by-life-cycle-stage.csv')
df = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')



## Have a look at some ingredients where ghg estimation was not possible
df[np.logical_and(df['ghg_missing']==1, # No GHG emission estimate
				  df['name_man_pruned2'].notna())] # Has GHG look-up label


##
## ++++++ Finding missing Units ++++++
##
## In most cases, when a unit is missing, it relates to single scallions,
## onions, bananas, etc.
items = df[np.logical_and(df['ghg_missing']==1, # No GHG emission estimate
				  df['name_man_pruned2'].notna())]['name_man_pruned1'].unique()
print(items)

## Try to find the most common unit based on the first pruned name and use as
## unit:
single_items = {}
items_not_found = []
for item in items:
	try:
		single_items[item] = df['unit_man'][df['name_man_pruned1'] == item].value_counts().keys()[0]
	except:
		items_not_found.append(item)

print('Could not assign units to the following ingredient labels (name_man_pruend1):')
print(items_not_found)

## Assign the missing ones manually:
## Listed in 'create_ingredient_dataframe.py
single_items['rice oil'] = df['unit_man'][df['name_man_pruned1'] == 'olive oil'].value_counts().keys()[0]
single_items['ball burrata'] = 'ounce'
single_items['maple leave'] = 'tablespoon'
single_items['corncob'] = 'cup'
single_items['cilantro oil'] = df['unit_man'][df['name_man_pruned1'] == 'olive oil'].value_counts().keys()[0]
single_items['soybean oil'] = df['unit_man'][df['name_man_pruned1'] == 'olive oil'].value_counts().keys()[0]
single_items['pheasant'] = 'pound'
single_items['locust'] = 'coffeespoon'
single_items['chips'] = 'cup'
single_items['barramundi'] = 'teacup'
single_items['raspberry'] = 'ounce'
single_items['giblet'] = 'pound'
single_items['bagel'] = 'teacup'
single_items['sweet potato'] = 'teacup'
single_items['taco'] = 'wineglass'
single_items['hominy'] = 'ounce'
single_items['matzoh'] = 'wineglass'
single_items['herring'] = 'teacup'
single_items['frico'] = 'teacup'
single_items['langoustine'] = 'teacup'
single_items['warp'] = 'teacup'
single_items['gingersnap'] = 'teacup'


## Some more fine grained adjustments:
df.loc[113449, 'qty_man'] = 6
df.loc[59073, 'qty_man'] = 3
df.loc[221028, 'qty_man'] = 2.75
df.loc[285549, 'qty_man'] = 2
df.loc[296729, 'qty_man'] = 2.25
df.loc[348169, 'qty_man'] = 2.25


## Remove keys that consist of numbers (comes from a mistake I made in 
## create_ingredient_dataframe.py - have to rerun)
number_keys = []
for key in single_items.keys():
	if key[0] in '0123456789':
		number_keys.append(key)
	print(key)

for delkey in number_keys:
	single_items.pop(delkey)


## Insert unit estimates to df
for key in single_items.keys():
	df.loc[df['name_man_pruned1']==key,'unit_man'] = single_items[key]



## Save updated data
df.to_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv')




## Let's have a look at the remaining rows without units
print('There are now', df[df['unit_man'].isnull()].shape[0], 'rows without units')
df[df['unit_man'].isnull()]


## The vast majority of these are spices
df[df['unit_man'].isnull()]['name_man_pruned1'].value_counts()

## As most spices probably do not contribute that much let's leave them for now







##
## ++++++ Finding missing qtys ++++++
##
## Try to find quantity based on comment and other
noq = df['qty_man'].isnull()
with_numbers = []
for i, (cmt, oth, inp) in enumerate(zip(df.loc[noq,'comment'], 
										df.loc[noq,'other'], 
										df.loc[noq,'input'])):
	if i % 4000 == 0:
		print(i, cmt, oth, inp)
		
	# Check for keywords in inp (for now do nothing)
	if 'to taste' in str(inp):
		pass
	if 'a pinch' in str(inp):
		pass
	# When there is no digit in any of the three columns do nothing
	if not any(char.isdigit() for char in str(cmt)+str(oth)+str(inp)):
		pass
	else:
		# When there are digits, keep track of index
		with_numbers.append(df.index[noq][i])
		
print('There are', len(with_numbers), 'rows with digits.')
df.loc[with_numbers, ['comment', 'other', 'input', 'unit_man','qty_man']]



## Go through rows with numbers and assign first number in input as quantity
## before space or dash i.e. 3 1/3 would be 3, 1/4- 1/3 would be 1/4.
qty_guesses = []
pattern1 = '[0-9\/\s-]+' 
#pattern2 = '[0-9\/]+[^\s]'
for i in with_numbers:
	string = df.loc[i]['input']
	if string[0:3] == 'one':
		qty_guesses.append('1')
	else:
		qty_guesses.append(re.search(pattern1, string, flags=0).group().strip())


# How do my guesses hold up?
for i, (guess, inp) in enumerate(zip(qty_guesses, df.loc[with_numbers, 'input'])):
	if i % 100 == 0:
		print(guess, '-->', inp)
	
	

## Convert guessed string to float
def parse_qty_text(s):
	## Dashes are used for ranges, always select the lower part
	s = s.replace('-', '')
	if len(s) == 0:
		return 0
	q = 0
	## Can we simply convert to float?
	try:
		q = float(s)
	except:
		## Case 1 example: 1 1/4 - compute 1+1/4
		if len(s) > 4:
			if s[0].isdigit() and s[1]==' ' and s[2].isdigit() and s[3]=='/' and s[4].isdigit():
				q = float(s[0])+float(s[2])/float(s[4])
		elif 1 <= len(s) <= 4:
			## Case 2 example: 1/8 - compute 1/8
			if s[0].isdigit() and s[1]=='/' and s[2].isdigit():
				q = float(s[0])/float(s[2])
			## Case 3 example: 1 28 - select 1
			if s[0].isdigit() and s[1]==' ':
				q = float(s[0])
		elif s == ' ':
			q = 0
		else:
			print('There is an exception for', s, '...')
			q = 0
	return q



## Convert guesses to numbers
qty_num = []
for guess in qty_guesses:
	qty_num.append(parse_qty_text(guess))


## Add to df
df.loc[with_numbers, 'qty_man'] = qty_num




## Update GHG emission estimates
def update_ghg(d, ghg):
	'''
	Parameters
	----------
	d : DataFrame
		ingredient data frame
	ghg : DataFrame
		mapping of ingredient categories to ghg emissions

	Returns
	-------
	LIST	   
	with ghg emissions where they could be estimated for each row in d
	'''
	ghg_list = list()
	for idx in d.index:
		
		if idx % 1000 == 0:
			print('Ingredient #', idx, 'of', d.shape[0])
		
		try:
			# convert "fractions" to decimals before converting str to float
			numstr = d.iloc[idx]['qty_man']
			if '/' in numstr:
				numL = numstr.split(sep='/')
				q = float(numL[0]) / float(numL[1])
			else:
				q = float(numstr)
		except:
			q = 0 # can I guess this instead?
		
		# convert quantity to ml (which is also roughly grams)
		try:
			qtotal = q * units_ml[d.iloc[idx]['unit_man']]
		except:
			qtotal = 0
			
		# Estimate GHG emissions
		ghg_val = qtotal * 0.001 * ghg['Total'][ghg['Food product'] == 
								   d.iloc[idx]['name_man_pruned2']].values
		
		if ghg_val.size == 0:
			ghg_val = 0
			
		ghg_list.append(float(ghg_val))
	return(ghg_list)




# Find ghg emission estimates for imputed values
ghg_list = update_ghg(df, ghg)


# Save data frame
df.to_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv')






# Load in again
df = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv')






## Clean up df - drop columns I don't need
## Check for each relevant column if there are nans and how many
## 
df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'range_end'], 
		inplace=True)

print('Number of missing values by column:')
df.isnull().sum()


## There are inputs with NaNs?
df.loc[df['input'].isnull(),] 
## Completely empty...


## What about name?
df.loc[df['name'].isnull(),]
## Should pretty much all be assignabl


## Drop rows based on NaN in input
df.dropna(subset=['input'], inplace=True)




## Have a look at some ingredients where ghg estimation was not possible
tmp = df.loc[df['name_man_pruned2'].isnull(),'name_man_pruned1'].value_counts()
for k, v in zip(tmp.keys(), tmp.values):
	print(k, v)
	
	
## Make a list of compound ingredients, that I can try to estimate. Single
## compound ingredients that are not currently assigned I will leave for the
## moment (salt, vanilla, baking soda...).
## This is all based on column 'name_man_pruned1'
# compound_ingredients = ['mayonnaise', 'worcestershire', 'dough', 'jam', 'pastry', 
# 						'vegan shortening', 'cookie', 'candy', 'hot sauce', 
# 						'cake', 'hoisin sauce', 'chimichurri sauce', 'salsa', 
# 						'margerine', 'barbecue sauce', 'custard', 'almond paste',
# 						'coriander chutney', 'brioche', 'miso', 'vinaigrette',
# 						'pesto', 'epazote', 'sriracha', 'garam masala', 
# 						'']

# Let's see if I can find these (some are compound ingredients) in the recipe
# names to estimate their GHG:
	
# Get recipe data frame
df_rec = pd.read_csv(r'D:\data science\nutrition\epi_recipes_clean.csv')



# Find matches
for missing_ingr in tmp.keys():
	# Go through recipes and find exact matches and contained matches 
	# (exact: mayonnaise, mayonnaise; contained: mayonnaise, sandwich with 
	# mayonnaise)
	recipe_in = []
	recipe_match = []
	for idx, rec in zip(df_rec.index, df_rec['title']):
		if missing_ingr == rec.lower().strip():
			recipe_match.append(idx)
		if missing_ingr in rec.lower().strip():
			recipe_in.append(idx)
			
	# If exact match exists (there can only be one), use that
	if recipe_match:
		est_ghg = 
		print('Found exact match for', missing_ingr, 'estimated ghg is', est_ghg)









## Try to find the most common unit based on the first pruned name and use as
## unit:
for 

## In some cases I might be able to infer the unit or quantity based on what
## is saved in comment, other or, indeed, in the original ingredient entry

## Start by looking for units in 
## 1.) comment,
## 2.) other,
## 3.) input


## Are there some where we can use the 'other' column to help?
df[np.logical_and(df.index.isin(ex), df['other'].notnull().values)]














































## EOF