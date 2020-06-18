# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:32:59 2020
	Use modules taken from the original Kaggle dataset to scrape recipes
	from the epicurious website:
	https://www.kaggle.com/hugodarwood/epirecipes?select=recipe.py
@author: sbuer
"""



# Add project folder to search path
import sys
sys.path.append(r'D:\data science\nutrition\scripts\tdi_challenge_may2020')

# Data management
import pickle
import json

# Check execution time
import time

# Import recipe modules from kaggle post
from recipes import EP_Recipe



# Load recipe links (from scrape_epicurious_links.py)
with open('epi_recipe_links', 'rb') as io:
    recipe_links = pickle.load(io)


# Convert recipe links to recipe objects (does 16 recipes in 10.97s)
print("Scraping recipes from epicurious.....") 
start_time = time.time()

ep_urls = ["https://www.epicurious.com" + i for i in recipe_links]
output = [EP_Recipe(url) for url in ep_urls[0:16]]

print("--- %s seconds ---" % (time.time() - start_time))
# for 34000 recipes it took 5.625 h

# Convert list of EP_Recipe objects to list of dictionaries
ar = []
for i in output:
	ar.append(i.__dict__)
	
# Dump to json
with open('epi_recipes_detailed', 'w') as io:
        json.dump(ar, io)






# eof