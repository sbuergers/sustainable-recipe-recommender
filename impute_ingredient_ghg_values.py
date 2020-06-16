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







##
## ++++++ Finding missing qtys ++++++
##
## Try to find quantity based on comment and other
for cmt, oth, inp in zip(df['comment'], df['oth'])









## Some more fine grained adjustments:
df.loc[113449, 'qty_man'] = 6
df.loc[59073, 'qty_man'] = 3
df.loc[221028, 'qty_man'] = 2.75
df.loc[285549, 'qty_man'] = 2
df.loc[296729, 'qty_man'] = 2.25
df.loc[348169, 'qty_man'] = 2.25







# Save data frame
df.to_csv(r'D:\data science\nutrition\ingredients_manually_processed2.csv')










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