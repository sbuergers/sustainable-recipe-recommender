# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:07:01 2020

Uses a json file with a list of dictionaries of recipe information from 
www.epicurious.com (created e.g. in scrape_epicurious_recipes.py)

@author: sbuer
"""

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



# Seems to be working really well. Do this for all data:
df_dummy = sublists_to_binaries(df,sublist) 

































# eof

