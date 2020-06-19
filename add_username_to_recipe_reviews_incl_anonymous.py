# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:36:20 2020

The following is done in "scrape_epicurious_recipe_reviews.py"

Scrape recipe reviews from epicurious website. Uses the data from
epi_r_w_sust.csv, which inlcudes recipe titles from the Kaggle dataset on
https://www.kaggle.com/hugodarwood/epirecipes

Reviews are saved in json format like so:
{'<title>':[
	{'review_text':Char,
     'rating':Int}
	]
}


This script adds username to the output dictionary, so:
	
{'<title>':[
	{'review_text':Char,
     'rating':Int
	 'username':Char}
	]
}

When no username can be determined an empty string is filled in

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




## (re-)load review data
with open('epi_reviews.txt') as json_file:
    review_dict = json.load(json_file)
	
## To do: Get user IDs out 

# Regular expression for finding username in review text
#
# () = group
# \.|\!|\?\) find ., !, ? or ) followed by \w+ ascii letters, followed by
# \s a separator like space, followed by from (should almost always work). 
#
# Example review text:
# "I made this salad and it was pretty good. However, instead of the mint 
# sugar I sprinkled plain sugar over it and put in whole mint leaves. I made 
# the mint sugar and it didn't taste right. I also left out the lemon and lime 
# and just squeezed the juice from them over the salad. I agree that it was 
# very time consuming, but it was a crowd pleaser. Paired really well with 
# sparkling wine.bainbridgebeck from Bainbridge Island WA"
re_username = "(\.|\!|\?\))\w+\sfrom"


# Go through reviews of each recipe and get out usernames
users = []
no_user_index = []
N_users_found = 0
N_users_total = 0
cook_num = 0
recipe_titles = review_dict.keys()
for i, title_raw in enumerate(recipe_titles):
	
	# Progress 
	if i % (len(recipe_titles)/25) == 0:
		print(i, title_raw)
	
	# Loop through reviews for this recipe
	for j, review in enumerate(review_dict[title_raw]):
		
		rating = review['rating']
		review_text = review['review_text']
		username_match = re.search(re_username, review_text)
		
		# if we did not match anything, append empty list (no user matched)
		if username_match is None:
			username = ""
			no_user_index.append((i,j))
		else:
			username = username_match.group(0).replace(' from', '').replace('.', '').lower()
			# if the username is A cook the user is anonymous
			if username == "a cook":
				username = "a_cook" + str(cook_num)
				cook_num += 1
				
		# Append username to dictionary
		review_dict[title_raw][j]['username'] = username
		
		# Keep track of how many reviews could be assigned to someone
		N_users_total = N_users_total + 1
		if len(username) > 1:
			N_users_found = N_users_found + 1

print('Found ', N_users_found, ' usernames out of ', N_users_total, ' reviews.')



# Save updated reviews dictionary to json
with open('epi_reviews_w_usernames_incl_anonymous.txt', 'w') as io:
    json.dump(review_dict, io)















## eof