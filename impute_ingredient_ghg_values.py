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


# Helper functions to convert json to dummy data frame
from categories_to_dummy import sublists_to_binaries, sublist_uniques




ghg = pd.read_csv(r'D:\data science\nutrition\GHG-emissions-by-life-cycle-stage.csv')
df = pd.read_csv(r'D:\data science\nutrition\ingredients_manually_processed.csv')

## Convert ghg column to float from string
ghg_float = np.zeros((df.shape[0],1))
for i in df.index:
	if len(df['ghg'][i]) > 2:
		ghg_float[i,] = float(df['ghg'][i][1:-1])
		
ghg_float = [float(df['ghg'][i][1:-1]) for i in df.index]

pd.to_numeric(df["ghg"], downcast="float")


## Define rows that do not have a valid ghg value
ex = df['ghg']>0
df[df['ghg'].astype(bool)]

## Have a look at some ingredients where ghg estimation was not possible
df[ex]


## Are there some where we can use the 'other' column to help?
df[np.logical_and(df.index.isin(ex), df['other'].notnull().values)]














































## EOF