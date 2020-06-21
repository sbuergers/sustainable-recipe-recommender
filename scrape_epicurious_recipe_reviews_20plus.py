# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:36:20 2020

In scrape_epicurious_recipe_reviews.py all recipe links (previously retrieved
scrape_epicurious_recipe_links.py) from the epicurious website 
(https://www.epicurious.com/) are scraped for user reviews. However, these
are missing all reviews beyond 20 (or 25?) per recipes. 

Here I am looking to find all additional recipe reviews, which can be found
by using selenium to click a "view more reviews" button at the bottom of the
webpage. This is very slow so it is prudent to do this only for recipes that
might have additional reviews.

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
	
	
def get_expanded_reviews_page(driver, url):
	"""Expands all recipe reviews of the given epicurious url by 'clicking' 
	the view more recipes button until it disappears. Returns html page. """
	## Connect to Epicurious recipe URL
	driver.get(url)
	
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
		timeout = 20
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
	review_texts = [rev.find('div', {'class': "review-text"}) for rev in reviews]
	reviews = [rev.get_text().strip('/ flag if inappropriate') for rev in review_texts]
	result = [
		{'review_text': review_text, "rating": rating_score}
		for review_text, rating_score in zip(reviews, ratings)
		]
	return result


# recipe-scrapers works beautifully for recipes with less than 25
# reviews. Here we are only looking at recipes with more than 25 reviews, 
# because using selenium to click the "load more reviews" button is slow. 


# Setup selenium webpage
driver = webdriver.Chrome()
time.sleep(10) # wait a few seconds for chrome to open


# Load recipe links (from scrape_epicurious_recipe_reviews.py)
with open('epi_reviews20200619_232923.txt', 'r') as io:
	reviews = json.load(io)

# Add "hidden" reviews where necessary
start_time = time.time()
N = len(reviews)
faillog = []
for i, url in enumerate(reviews.keys()):
	
	#print(i, url, len(reviews[url]))

	if len(reviews[url]) > 24:
		
		print('Adding new reviews:', i, url, len(reviews[url]))
		
		try:
			# Get html text of full page (with all reviews)
			webpart = 'https://www.epicurious.com/recipes/food/views/'
			page = get_expanded_reviews_page(driver, webpart + url)
	
			# scrape reviews from recipe page
			page_reviews = get_reviews(page)
		
			# Update review dictionary with additional reviews
			reviews[url] = page_reviews
		except:
			faillog.append([i, url])

# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Save reviews dictionary to json
with open('epi_reviews_25plus.txt', 'w') as io:
    json.dump(review_dict, io)











##############################################################################
## Code snippets

# ## Selenium code for infinite scroll (must be slow though!)
# # Add project folder to search path
# import sys
# sys.path.append(r'D:\data science\nutrition\scripts\tdi_challenge_may2020')


# # to get additional recipes I need to either "click" next page or virtually
# # scroll down for more recipes to load
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import os

# browser = webdriver.Chrome(executable_path=os.path.join(os.getcwd(),'chromedriver'))
# browser.get(search_url)

# body = browser.find_element_by_tag_name("body")
# browser.Manage().Window.Maximize(); 

# no_of_pagedowns = 2 #Enter number of pages that you would like to scroll here

# while no_of_pagedowns:
#     body.send_keys(Keys.PAGE_DOWN)
#     no_of_pagedowns-=1




# options = webdriver.ChromeOptions()
# options.add_argument('--ignore-certificate-errors')
# options.add_argument("--test-type")
# options.binary_location = "/usr/bin/chromium"
# driver = webdriver.Chrome(chrome_options=options)
# driver.get('http://codepad.org')

# # click radio button
# python_button = driver.find_elements_by_xpath("//input[@name='lang' and @value='Python']")[0]
# python_button.click()

# # type text
# text_area = driver.find_element_by_id('textarea')
# text_area.send_keys("print('Hello World')")

# # click submit button
# submit_button = driver.find_elements_by_xpath('//*[@id="editor"]/table/tbody/tr[3]/td/table/tbody/tr/td/div/table/tbody/tr/td[3]/input')[0]
# submit_button.click()



## eof



