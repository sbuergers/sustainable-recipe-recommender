# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:40:47 2020
	Use modules taken from the original Kaggle dataset to scrape recipes
	from the epicurious website:
	https://www.kaggle.com/hugodarwood/epirecipes?select=recipe.py
	
	Here I do not search for all recipes in one go, but search for specific
	recipe categories (e.g. vegan, paleo, etc.). The purpose is potentially
	two-fold: 1.) I want to see if I get more recipes than by searching 
	without a search term; and 2.) I want to be able to assign new categories
	if I think there are some missing from the recipes I currently have. 
@author: sbuer
"""

# Package for scraping recipes from many popular websites, for details see 
# https://github.com/sbuergers/recipe-scrapers/blob/master/recipe_scrapers/epicurious.py
from recipe_scrapers import scrape_me

# Get HTML from website
import requests

# Regular expressions
import re

# Input / output
import pickle

# Check execution time
import time



# Main function - gets all recipe links for a specific search term from 
# epicurious and appends them to pickle file
def scrape_epi_links(search_term):

	# URL of epicurious search for newest recipes:
	# 'https://www.epicurious.com/search/vegan?content=recipe&sort=newest'
	url_start = r'https://www.epicurious.com/search/'
	url_end = r'?content=recipe&sort=newest'
	initial_search_url = url_start + search_term + url_end
	
	# After the first page the url also includes the page number as follows:
	# https://www.epicurious.com/search?content=recipe&page=2&sort=newest
	
	# scrape search url and get HTML text
	page = requests.get(initial_search_url)
	html_text = page.content.decode('utf-8')
	
	# find recipe urls and collect unique recipe links in list
	# Example: href="/recipes/food/views/spring-chicken-dinner-salad" 
	re_rec = r"\/recipes\/food\/views\/(\w+|\-)+" 
	recipe_links = list(set([x.group() for x in re.finditer(re_rec, html_text)]))
	
	# Go through additional recipes by increasing the page number in the urlimport time
	start_time = time.time()
	pagenum = 2
	while True:
	#for i in range(0,10): # try with for-loop first for testing
	
		# progress
		if pagenum % 10 == 0:
			print("Page #", pagenum, "Number of recipes scraped = ", len(recipe_links))
	
		# get next recipe page in HTML
		search_url = url_start + search_term + "?content=recipe&page={}&sort=newest".format(pagenum)
		page = requests.get(search_url)
		
		# stop looking when max page number is reached
		if page:
			html_text = page.content.decode('utf-8')
			pagenum += 1
			
			# collect recipe links and append to list
			more_links = list(set([x.group() for x in re.finditer(re_rec, html_text)]))
			recipe_links += more_links
		else:
			print("Reached bottom of page")
			break
	print("--- %s seconds ---" % (time.time() - start_time))
	
	# Make sure recipe links are truly unique (should already be)
	recipe_links = list(set(recipe_links))
	
	return recipe_links
	





# # Load recipe links from pickle file
# with open('epi_recipe_links', 'rb') as io:
#     # read the data as binary data stream
#     recipe_links = pickle.load(io)


# Scrape recipe links by category (appends to pickle file)
# Try out for one search term to test
search_term = r'Vegan'
recipe_links = scrape_epi_links(search_term)


# Do I have all of these also in the non-directed search list of recipe links?

# Load in previously scraped data 
with open('epi_recipes_detailed', 'r') as io:
        data = json.load(io)
df = pd.DataFrame(data)

# Check how many recipes I already had collected
old_links = df['url'].values
new_links = [l[len('/recipes/food/views/')::] for l in recipe_links]
matches = [n in old_links for n in new_links]

print('We already had', sum(matches), 'out of', len(recipe_links), 'recipe links.')


# It's proboably save to say that the few additional ones are simply new.
# So we can conclude that searching for specific epicurious categories does
# not yield recipe links that we did not get with a simple unspecified search.

# The question remains: Why does epicurious claim to have 300k+ recipes?
# I can only find 40k- searching their website...


# Save recipe links to txt file
#with open('epi_recipe_links_by_category', 'a') as io:
#    pickle.dump(recipe_links, io)




# eof
















