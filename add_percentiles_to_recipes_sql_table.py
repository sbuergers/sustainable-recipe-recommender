# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:12:28 2020

Add percentiles to recipes table on AWS RDS postgres DB,
also add image_url column to recipes table

@author: sbuer
"""


# for loading environment variables
import os

# needed to load environment variables from .env file
from dotenv import load_dotenv 
load_dotenv()

# for connecting to postgres database
import psycopg2 as ps
from psycopg2 import sql

# pandas and numpy
import pandas as pd
import numpy as np


# Load recipes data (which I used as .csv before)
recipes = pd.read_csv(r'D:/data science/nutrition/data/recipes_sql.csv', index_col=0)


# create connection and cursor for AWS RDS postgres DB db-srr
conn = ps.connect(host=os.environ.get('AWS_POSTGRES_ADDRESS'),
                  database=os.environ.get('AWS_POSTGRES_DBNAME'),
                  user=os.environ.get('AWS_POSTGRES_USERNAME'),
                  password=os.environ.get('AWS_POSTGRES_PASSWORD'),
                  port=os.environ.get('AWS_POSTGRES_PORT'))
cur = conn.cursor()


# create pandas dataframe from recipes table on AWS RDS postgres DB
cur.execute("""
			SELECT "recipesID", "title", "ingredients",
				   "rating", "calories", "sodium", "fat",
				   "protein", "emissions", "prop_ingredients", 
				   "emissions_log10", "url", "servings", "recipe_rawid"
		    FROM public.recipes
			""")
sql_output = cur.fetchall()

sel = ['recipesID', 'title', 'ingredients', 'rating', 'calories', 'sodium', 
	   'fat', 'protein', 'ghg', 'prop_ing', 'ghg_log10', 'url', 'servings', 
	   'index']
recipes_sql = pd.DataFrame(sql_output, columns=sel) 


## The image_url is not part of the sql database yet, add that column
recipes_new = recipes_sql.merge(recipes['image_url'], on='recipesID', how='left')


## Next I need to add columns for percentiles of ratings and sustainability
def rating_to_percentage(results):
    '''
    DESCRIPTION:
        Converts raw ratings to percentage ratings (e.g. 4
        stars would be converted to 80%)
    INPUT:
        results (DataFrame): Subset of recipes DataFrame containing
            similar recipes to reference recipe to be diplayed
			
    OUTPUT:
        rating_percentage (list): List of length(results) with the
            corresponding percentage ratings
    '''
    # Replace nans with 0
    results['rating'] = results['rating'].fillna(0)
    return [round(ra*20) for ra in results['rating']]


def sustainability_to_percentage(results):
    '''
    DESCRIPTION:
        Converts raw emissions scores to quantiles
    INPUT:
        results (DataFrame): Subset of recipes DataFrame containing
            similar recipes to reference recipe to be diplayed
    OUTPUT:
        sustainability score (list): 100 - emission-quantile
    '''
    N = results.shape[0]
    emission_quantile = [round(sum(results['ghg'] < em)*100 / N)
                         for em in results['ghg']]
    return [100 - eq for eq in emission_quantile]


## Add columns for percentiles for

## 1.) rating
recipes_new['perc_rating'] = rating_to_percentage(recipes_new)

## 2.) sustainability
recipes_new['perc_sustainability'] = sustainability_to_percentage(recipes_new)


## Finally, add columns from recipes_sql.csv that I couldn't load from the DB,
## and reorder to fit recipes SCHEMA
recipes_extended = recipes_new.merge(recipes[['categories', 'date', 'review_count']],
									on='recipesID', how='left') 
recipes_extended = recipes_extended[[
	'recipesID', 'title', 'ingredients', 'categories', 'date', 
	'rating', 'calories', 'sodium',
	'fat', 'protein', 'ghg', 'prop_ing', 'ghg_log10', 'url', 'servings',
    'index', 'image_url', 'perc_rating', 'perc_sustainability', 'review_count']]


## I have whitespaces in character columns from previous SQL export, delete
def strip_whitespace(recipes_extended, colname):
	return [v.strip() if type(v)==str else '' for v in recipes_extended[colname].values]

string_columns = ['title', 'ingredients', 'categories', 'url', 'servings',
				  'image_url']
for sc in string_columns:
	recipes_extended[sc] = strip_whitespace(recipes_extended, sc)


## Export data to .csv file and upload using pgadmin4
## (probably faster than using psycopg2). Note index=False.
recipes_extended.to_csv(r'D:\data science\nutrition\data\recipes_sql_28082020.csv',
						index=False)


# eof











