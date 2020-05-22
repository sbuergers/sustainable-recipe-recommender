# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:02:51 2020

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



# URL of epicurious search for newest recipes:
initial_search_url = r"https://www.epicurious.com/search/?content=recipe&sort=newest"

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
	search_url = r"https://www.epicurious.com/search?content=recipe&page={}&sort=newest".format(pagenum)
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


# Save recipe links to txt file
with open('epi_recipe_links', 'wb') as io:
    # store the data as binary data stream
    pickle.dump(recipe_links, io)




# # Load recipe links from pickle file
# with open('epi_recipe_links', 'rb') as io:
#     # read the data as binary data stream
#     recipe_links = pickle.load(io)






# eof