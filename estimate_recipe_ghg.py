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




## I should probably plug these two into a SQL databse...

# Load recipe data
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
print(df_ing.loc[df_ing['recipe_id'] == randIdx, 'input'])
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








# eof