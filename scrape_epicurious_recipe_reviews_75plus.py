# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:36:20 2020

Sometimes, scrape_epicruious_recipe_reviews_25plus.py did not manage to
get all reviews beyond 25. Here I try to get them, assuming that it always 
worked when there were fewer than 76 reviews (which requires only 2 button
presses by selenium). 

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

# Check for files / paths
import os.path
from os import path

# Data management
import pandas as pd 
import json
import pickle

# Check execution time
import time

# parsing page (scrape_me wants url, not text of page)
from bs4 import BeautifulSoup as bs

# Get selenium to "press" load more recipes button (there should be an easier
# way to do this, but not sure how)
## From 
## https://codereview.stackexchange.com/questions/169227/scraping-content-from-a-javascript-enabled-website-with-load-more-button
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


def get_load_reviews_button(driver):
	"""Returns the load more reviews button element if it exists"""
	try:
		return driver.find_element(By.XPATH, '//button[text()="View More Reviews"]')
	except NoSuchElementException:
		return None
	
	
def center_page_on_button(driver, button):
	"""Gets the load more reviews button into view (so it's clickable) """
	try:
		if button:
			driver.execute_script("arguments[0].scrollIntoView();", button)
			driver.execute_script("window.scrollBy(0, -150);")
	except:
		raise
	
	
def click_load_reviews_button(button):
	"""Attemps to hover over and click the load more views button """
	try:
		button.click()
# 		hover = ActionChains(driver).move_to_element(button)
# 		hover.perform()
# 		button.click()
		return "button_clicked"
	except StaleElementReferenceException:
		return "no_button"
	except AttributeError:
		return "no_button"
	except ElementClickInterceptedException:
		return "pop_up_interferes"
	except:
		raise
		
		
def close_pop_up(driver):
	"""Makes selenium 'press' the ESC key to close pop-up window """
	webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
	
	
def get_expanded_reviews_page(driver, fullurl):
	"""Expands all recipe reviews of the given epicurious url by 'clicking' 
	the view more recipes button until it disappears. Returns html page. """
	## Connect to Epicurious recipe URL
	driver.get(fullurl)
	
	# Do we have a load more reviews button?
	button = get_load_reviews_button(driver)
	
	# If so, attempt to click the Load Reviews Button until it vanishes
	if button:
		
		# center page on load more reviews button
		center_page_on_button(driver, button)
		
		# click the button
		status = click_load_reviews_button(button)
		
		# Keep doing this until the button disappears or we time out with an error
		start_time = time.time()
		run_time = 0
		timeout = 120
		while (button) and (not status == "no_button") and (run_time < timeout):
			if status == "pop_up_interferes":
				close_pop_up(driver)
			button = get_load_reviews_button(driver)
			center_page_on_button(driver, button)
			status = click_load_reviews_button(button)
			run_time = time.time()-start_time
			
	return driver.page_source


## Since recipe scrapers internally uses requests and only takes url as input,
## rather than rewriting the toolbox to also accept page content, adapt the 
## function that gets users reviews and include it here:
def get_reviews(page):
	"""Scrapes review texts from epicurious web-pages. Page is the HTML of the
	web-page. result is a dictionary with 'review_text' and 'rating' as keys,
	including a string and integer as values, respectively."""
	fork_rating_re = re.compile('/(\d)_forks.png')
	soup = bs(page, 'html.parser')
	reviews = soup.findAll('', {'class': "most-recent"})
	ratings = [rev.find('img', {'class': "fork-rating"}) for rev in reviews]
	temp = []
	for rating in ratings:
		if 'src' in rating.attrs:
			txt = rating.attrs['src']
		else:
			txt = ''
		rating = fork_rating_re.search(txt)
		rating = rating.group(1) if rating is not None else '0'
		rating = int(rating) if rating != '0' else None
		temp.append(rating)
		ratings = temp
	review_text = [rev.find('div', {'class':'review-text'}).p.text for rev in reviews]
	review_sign = [rev.find('span', {'class': 'credit'}).text for rev in reviews]
	review_sign = [rev[:-3] for rev in review_sign]
	result = [
		{'review_text': text, 'rating': score, 'signature': sign}
		for text, score, sign in zip(review_text, ratings, review_sign)
		]
	return result


# Setup selenium webpage
# Includes adding adblock extension and skipping loading of images
# NOTE: Occasionally restarting the driver speeds the process up tremendously!
def initialize_selenium_session():
	"""Initiates a selenium chrome session without loading images and an 
	adblock extension."""
	prefs = {'profile.managed_default_content_settings.images': 2} 
	chrome_options = webdriver.ChromeOptions()
	chrome_options.add_extension(r'D:\data science\nutrition\misc\AdBlockPlus.crx') 
	chrome_options.add_experimental_option('prefs', prefs) 
	driver = webdriver.Chrome(options=chrome_options)
	time.sleep(10) # wait a few seconds for chrome to open
	return driver


# recipe-scrapers works beautifully for recipes with less than 25
# reviews. Here we are only looking at recipes with more than 25 reviews, 
# because using selenium to click the "load more reviews" button is slow. 


# Load recipe links (from scrape_epicurious_recipe_reviews_from_faillog.py)
with open('epi_reviews_25plus_final.txt', 'r') as io:
	reviews = json.load(io)
	
# Initialize Selenium browser session
driver = initialize_selenium_session()

# Add "hidden" reviews where necessary
start_time = time.time()
faillog = []
reviews_new = {}
for i, url in enumerate(reviews.keys()):
	
	# Only run over a subset (e.g. already did the first 5000):
	if i < 33000: # change manually!
		continue
	
	if len(reviews[url]) > 75:
		
		# Sometimes it simply does't work, retry a few times, otherwise
		# remember where it failed
		num_tries = 0
		no_success = True
		while (num_tries < 5) and (no_success):
			try:
				# Get html text of full page (with all reviews)
				webpart = r'https://www.epicurious.com/recipes/food/views/'
				page = get_expanded_reviews_page(driver, webpart + url)
		
				# scrape reviews from recipe page
				page_reviews = get_reviews(page)
			
				# Update review dictionary with additional reviews
				reviews[url] = page_reviews
				no_success = False
			except:
				num_tries += 1
		if num_tries == 5:
			faillog.append([i, url])
			
		print('Adding new reviews:', i, url, len(reviews[url]))
	
	# Save periodically
	reviews_new[url] = reviews[url]
	if ((i+1) % 200 == 0) | (i==len(reviews)):
		
		# Saving dictionaries is a bit of a pain if done recurrently,
		# but I can simply load in the previous dictionary and append
		if path.exists('epi_reviews_75plus.txt'):
			with open('epi_reviews_75plus.txt', 'r') as io:
				reviews_old = json.load(io)
			reviews_to_file = {**reviews_old, **reviews_new}
		else:
			reviews_to_file = reviews_new
		
		# Save reviews dictionary to json
		with open('epi_reviews_75plus.txt', 'w') as io:
			json.dump(reviews_to_file, io)
		reviews_new = {}
		
		# Write fail-log to file 
		with open('epi_reviews_75plus_faillog.txt', 'a') as io:
			for item in faillog:
				io.write('%s\n' % item)
		faillog = []
		
		print('\n ----- Saving to file ----- \n')
		
	# As Chrome slows down over time, it makes sense to periodically restart
	# the Selenium session (i.e. close and restart)
	if (i+1) % 1000 == 0:
		driver.quit()
		driver = initialize_selenium_session()
		
# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Tidy up Selenium browser session
driver.quit()





#######
## Test if adding additional reviews worked:
	
# Load old data (max 25 reviews per recipe)
with open('epi_reviews_25plus_final.txt', 'r') as io:
	reviews = json.load(io)
# Load new data
with open('epi_reviews_75plus.txt', 'r') as io:
	reviews_25plus = json.load(io)
tot_diff = 0
tot_neg = 0
n_diff = 0
print('Idx | Cn | Co | Cn-Co')
for idx, (i, j) in enumerate(zip(reviews_25plus.values(), reviews.values())):
	if len(i) > len(j):
		tot_diff += len(i)-len(j)
	elif len(j) > len(i):
		tot_neg += len(j)-len(i)
	if len(i) != len(j):
		n_diff += 1
		print(idx, len(i), len(j), len(i)-len(j))
print('There are', tot_diff, 'more reviews after updating, coming from', n_diff, 'recipes.')
print('There are also at least', tot_neg, 'fewer recipes, because of failed attempts.')

# Failed attempts result in fewer recipes than there actually are, this could
# even be zero, hence the lower bound of missed reviews in the 2nd print stmt.


## eof



