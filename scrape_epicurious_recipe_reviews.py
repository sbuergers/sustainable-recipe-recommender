# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:36:20 2020

Scrape recipe reviews from epicurious website. Uses the data from
epi_r_w_sust.csv, which inlcudes recipe titles from the Kaggle dataset on
https://www.kaggle.com/hugodarwood/epirecipes

Reviews are saved in json format like so:
{'<title>':[
	{'review_text':Char,
     'rating':Int}
	]
}

@author: sbuer
"""

# Package for scraping recipes from many popular websites, for details see 
# https://github.com/sbuergers/recipe-scrapers/blob/master/recipe_scrapers/epicurious.py
from recipe_scrapers import scrape_me

# Get HTML from website
import requests

# Regular expressions
import re

# Data management
import pandas as pd 
import json
import pickle

# Check execution time
import time




## Change some defaults for visualizing data frames
pd.set_option("max_columns", 15)
pd.set_option("max_rows", 15)



# Load recipe links (from scrape_epicurious_links.py)
with open('epi_recipe_links', 'rb') as io:
    recipe_links = pickle.load(io)

ep_urls = ["https://www.epicurious.com" + i for i in recipe_links]


# recipe-scrapers works beautifully if I have the url for the specific recipe
start_time = time.time()
review_dict = {}
N = len(df_rec)
for i, url in enumerate(ep_urls):
	
	# Progress 
	if i % 100 == 0:
		print(i, url)
		
	# Give the server some rest every 500 recipes
# 	if i % 500 == 0:
# 		time.sleep(60) # in s

	# scrape reviews from recipe page
	scraper = scrape_me(url)
	reviews = scraper.reviews()

	# Add recipe to review dictionary
	webpart = 'https://www.epicurious.com/recipes/food/views/'
	pruned_url = url[len(webpart)::]
	review_dict[pruned_url] = reviews

# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Save reviews dictionary to json
with open('epi_reviews.txt', 'w') as io:
    json.dump(review_dict, io)



## eof



