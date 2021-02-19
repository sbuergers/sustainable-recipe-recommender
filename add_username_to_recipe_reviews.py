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
with open(r'D:\data science\nutrition\epi_reviews_75plus.txt') as json_file:
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
re_user_template1 = "(\.|\!|\?|\))([\s]+)?([\r\n]+)?\w+\s+from"
# "(\.|\!|\?|\))([\s]+)?([\r\n]+)?\w+\sfrom" 
# "(\.|\!|\?\))\w+\sfrom"
re_user_template2 = "(\.|\!|\?\))[\s+]?\w+[\s+]?[\w+]?[\s+]?[\w+]?$"

# User template examples:
#
# re_user_template1 (with from):
#
# Good but not spectacular. The dressing didn't have enough flavor to hold up 
# to the pearled barley -- I upped the basil and that helped. It holds in the 
# fridge for a couple of days without getting soggy, which is good because 
# this recipe makes a lot of salad!tastysausagerecipe from Brookly

# re_user_template2 (without from):
#
# Loved it, and so did\nthe hubby. Bright,\nsummery, quick and\neasy. We made 
# it\nwith barley. I\nalways love a chance\nto use one of those\ngrains 
# sitting in my\npantry that I have a\nhard time thinking\nwhat to do 
# with.mrsrogers


# Go through reviews of each recipe and get out usernames
users = []
anonymous_index = []
n_rev = 0
cook_num = 0
recipe_titles = review_dict.keys()
for i, title_raw in enumerate(recipe_titles):
	
	# Progress 
	if i % (len(recipe_titles)/25) == 0:
		print(i, title_raw)
	
	# Loop through reviews for this recipe
	for j, review in enumerate(review_dict[title_raw]):
		
		n_rev += 1
		
		rating = review['rating']
		review_text = review['review_text']
		
		# Try to find username with template1
		username_match = re.search(re_user_template1, review_text)
		if not username_match is None:
			username = username_match.group(0).replace(' from', '').replace('.', '').lower()
			
		# if we did not match anything with template1, try template2
		elif username_match is None:
			username_match = re.search(re_user_template2, review_text)
			if not username_match is None:
				username = username_match.group(0).replace('.', '').strip().lower().split(' ')[0]
			
		# if we did not match anything, append unique ID (anonymous match)
		if (username_match is None) | (username == "a cook"):
			username = "a_cook" + str(cook_num)
			cook_num += 1
			anonymous_index.append((i,j))
				
		# Append username to dictionary
		review_dict[title_raw][j]['username'] = username
		users.append(username)


n_users = len(set(users))
print('Found', n_users, 'usernames out of', n_rev, 'reviews. Out of these', 
	  len(anonymous_index), 'were anonymous reviews.')


# Save updated reviews dictionary to json
with open(r'D:\data science\nutrition\epi_reviews_75plus_w_usernames.txt', 'w') as io:
    json.dump(review_dict, io)



## eof