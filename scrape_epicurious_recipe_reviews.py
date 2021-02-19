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


# recipe-scrapers works beautifully if I have the url for the specific recipe.
# However, it does not read reviews beyond what is shown on the website by
# default. To retrieve reviews hidden by the "view more reviews" button,
# I can use selenium. This is implemented in a different script, which only
# considers the recipes found here, with more than the feault number of 
# reviews. See scrape_epicurious_recipe_reviews_25plus.py.
start_time = time.time()

# Set filename
timestr = time.strftime("%Y%m%d_%H%M%S") # make filename unique for every run
filename = 'epi_reviews' + timestr + '.txt'

# Go through all files that are not already in filename
try:
	with open(filename, 'r') as io:
		old_reviews = json.load(io)
	S = len(old_reviews.keys())
except:
	S = 0
N = len(ep_urls)

review_dict = {}
for i, url in enumerate(ep_urls[S:N]):

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
	
	# Progress 
	if i % 100 == 0:
		print(i, url)	

# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Save reviews dictionary to json (append every 1000 recipes)
with open(filename, 'w') as io:
    json.dump(review_dict, io)
	

## eof



