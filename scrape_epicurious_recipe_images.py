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
from recipe_scrapers import scrape_me



# Load recipe links (from scrape_epicurious_links.py)
with open('epi_recipe_links', 'rb') as io:
    recipe_links = pickle.load(io)

ep_urls = ["https://www.epicurious.com" + i for i in recipe_links]


print("Scraping recipe image urls from epicurious.....") 
start_time = time.time()

# Sometimes it gets stuck - retrieve recipes in batches and save periodically
output = []
for i, url in enumerate(ep_urls):
	
	# progress
	if i % 100 == 0:
		print(i, url)
	
	# give the url as a string, it can be url from any site listed below
	scraper = scrape_me(url)
	output.append(scraper.image())

print("--- %s seconds ---" % (time.time() - start_time))
# for 34000 recipes it took 5.625 h





# save to file
with open('epi_recipe_images', 'wb') as io:
    # store the data as binary data stream
    pickle.dump(output, io)




# Load recipe links from pickle file
with open('epi_recipe_images', 'rb') as io:
    # read the data as binary data stream
    recipe_images = pickle.load(io)


# merge with recipes data frame
images = pd.DataFrame({'image_url': recipe_images})

print('There are', len(images.dropna()), 'recipes with pictures.')

recipes = pd.read_csv(r'D:/data science/nutrition/data/recipes_sql.csv', index_col=0)
recipes = recipes.merge(images, left_on='index', right_index=True)



recipes.to_csv(r'D:/data science/nutrition/data/recipes_sql.csv')








# eof
