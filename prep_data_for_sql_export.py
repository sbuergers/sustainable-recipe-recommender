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



df_rec.to_csv(r'D:\data science\nutrition\epi_recipes_with_ghg.csv')














