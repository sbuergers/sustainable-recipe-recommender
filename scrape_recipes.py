# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:36:20 2020

@author: sbuer
"""

# for details see 
# https://github.com/sbuergers/recipe-scrapers/blob/master/recipe_scrapers/epicurious.py
from recipe_scrapers import scrape_me

# give the url as a string, it can be url from any site listed below
scraper = scrape_me('https://www.epicurious.com/recipes/food/views/braised-chicken-with-artichokes-and-olives-51150800')

scraper.title()
scraper.total_time()
scraper.yields()
scraper.ingredients()
scraper.instructions()
scraper.image()
scraper.links()
scraper.reviews()










## eof