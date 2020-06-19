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
import pickle

# Check execution time
import time

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
	driver.get('https://www.epicurious.com/recipes/food/views/braised-chicken-with-artichokes-and-olives-51150800')
	
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



# recipe-scrapers works beautifully for recipes with less than 20 (or now 25?)
# reviews. Here we are only looking at recipes with more than 20 reviews, 
# because using selenium to click the "load more reviews" button is slow. 

# Load recipe links (from scrape_epicurious_recipe_reviews.py)
with open('epi_reviews.txt', 'r') as io:
	reviews = json.load(io)


# recipe-scrapers works beautifully if I have the url for the specific recipe
start_time = time.time()
review_dict = {}
N = len(df_rec)
for i, url in enumerate(reviews.keys()):
	
	# Progress 
	if i % 100 == 0:
		print(i, url)
		
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

# Code timing
print("--- %s seconds ---" % (time.time() - start_time))


# Save reviews dictionary to json
with open('epi_reviews_20plus.txt', 'w') as io:
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



