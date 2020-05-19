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

# Check execution time
import time


## Change some defaults for visualizing data frames
pd.set_option("max_columns", 15)
pd.set_option("max_rows", 15)



# Load recipe data frame 
df_rec = pd.read_csv("epi_r_w_sust.csv")



# recipe-scrapers works beautifully if I have the url for the specific recipe
# To get it I will use the search functionality of epicurious putting in the
# recipe's title. 
# For example:
# https://www.epicurious.com/search/braised-chicken-with-artichokes-and-olives?search=braised-chicken-with-artichokes-and-olives
# Then I will simply look for the recipe handle in the HTML corpus to get the
# recipe's specific link. 
start_time = time.time()
review_dict = {}
no_link_index = list()
N = len(df_rec)
for i, title_raw in enumerate(df_rec['title'][0:N]):
	
	# Progress 
	if i % 10 == 0:
		print(i, title_raw)
		
	# Give the server some rest every 500 recipes
# 	if i % 500 == 0:
# 		time.sleep(60) # in s
	
	# Remove commas and lagging spaces, replace spaces inbetween words with -,
	# and make lower case
	title = title_raw.strip().replace(',', '').replace(' ', '-').lower()
	
	# create recipe search url and scrape HTML text
	rec_search_url = "https://www.epicurious.com/search/" + title + "?" + "search=" + title
	page = requests.get(rec_search_url)
	html_text = page.content.decode('utf-8')
	
	# Get recipe url handle (including number at end) 
	find_me = title + "-" + "\d+"
	re_search = re.search(find_me, html_text)
	if re_search is None:
		reviews = []
		no_link_index.append(i)
	else:
		rec_handle = re_search.group(0)
	
		# create url of recipe
		rec_url = 'https://www.epicurious.com/recipes/food/views/' + rec_handle
		
		# scrape reviews from recipe page
		scraper = scrape_me(rec_url)
		reviews = scraper.reviews()
	
	# Add recipe to review dictionary
	review_dict[title_raw] = reviews

# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Save reviews dictionary to json
with open('epi_reviews.txt', 'w') as io:
    json.dump(review_dict, io)












## eof