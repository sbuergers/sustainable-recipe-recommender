# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:30:01 2020

Scrape tweets using GetOldTweets3 related to recipes in databse (for now only
epicurious recipes from Kaggle). 

Kaggle link: https://www.kaggle.com/hugodarwood/epirecipes

@author: sbuer
"""

# Data science libraries
import numpy as np
import pandas as pd 

# Handle sparse matrices efficiently
from scipy.sparse import csr_matrix

# Saving and loading
import json

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

## Twitter 
import GetOldTweets3 as got # archived tweets

## Make python wait for twitter server 
import time




## Load recipe data
df_rec = pd.read_csv('epi_r_w_sust.csv')




## Search for recipe names and 'epicurious' (do not allow retweets)
## Runs into rate limit fairly quickly (after around 400-500 recipes), so take
## periodic breaks
#compl_dict = {}
#compl_dict['tweets'] = []
#user_dict = {}
cnt = 0
for i, title in enumerate(df_rec['title'][0:]):
	
	## take a timeout every 500 searches 
	cnt = cnt + 1
	if cnt > 500:
		time.sleep(900)
		cnt = 0
	if i % 100 == 0:
		print(i, title)
		
	## search for recipe tweets
	search_words = title + '#epicurious' + ' -RT'
	max_tweets = 100
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(search_words)\
		.setSince("2010-01-01")\
		.setUntil("2020-05-05")\
		.setMaxTweets(max_tweets)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	
	## if tweets is not empty, add users with title to dictionary
	if tweets:
		print(i, title, ': Tweet found!!!')
		## Save whether recipe was tweeted by specific user (concise)
		for j, tweet in enumerate(tweets):
			user = tweet.username
			if user in user_dict:
				user_dict[user].add(title)
			else:
				user_dict[user] = set([title])
		
			## Save every tweet with user, date and location information in dict
			## including duplicate tweets and dump to json
			compl_dict['tweets'].append({
				'name':user, 
				'text':tweet.text,
				'permalink':tweet.permalink,
				'to':tweet.to,
				'hashtags':tweet.hashtags,
				'location':tweet.geo})


## Write comprehensive tweet information to json
with open('user_recipe_tweets.txt', 'w') as f:
    json.dump(compl_dict, f)


## If you wanted to load this back in:
## with open('user_recipe_tweets.txt') as f:
## 	test = json.load(f)


## I do not have to construct a user x recipe title matrix, it is more efficient
## to save everything in a data frame with user and recipe title as columns,
## which is what the ML functions in python are used to. 
user_keys = user_dict.keys()
user_list = list()
title_list = list()
for user in user_keys:
	user_titles = list(user_dict[user])
	for title in user_titles:
		user_list.append(user)
		title_list.append(title)

## Construct data frame and save to csv file
df_users = pd.DataFrame({'user':user_list, 'title':title_list})
print('Number of users: ', len(user_keys))
df_users.to_csv('epi_users.csv')




# Create sparse matrix of users x recipes with "ratings" as cells, i.e. for
# now this is just whether the recipe was tweeted or not. 

users = df_users['user'].unique()
titles = df_users['title'].unique()
Nx = users.shape[0]
Ny = titles.shape[0]
user_rec_mat = csr_matrix((Nx, Ny), dtype=np.int8)
for u, user in enumerate(users):
	user_titles = list(df_users['title'][df_users['user'] == user])
	match_ids = [j for j, val in enumerate(titles) if val in user_titles] 
	user_rec_mat[u, match_ids] = 1

user_rec_mat
plt.imshow(user_rec_mat.toarray())
plt.colorbar()
plt.show()





## Compute number of recipe tweets by user
N_tw_by_user = np.sum(user_rec_mat.toarray(), axis=1)

## plot distribution of number of recipe tweets per user
fig, ax = plt.subplots(figsize=(10,10))
ax.hist(N_tw_by_user, bins=50)
ax.set_ylabel('Number of recipe tweets')
ax.set_title('Distribution of number of recipe tweets by users')
plt.show()




## Compute number of tweets by recipe
N_tw_by_recipe = np.sum(user_rec_mat.toarray(), axis=0)

## plot distribution of number of tweets per recipe
fig, ax = plt.subplots(figsize=(10,10))
ax.hist(N_tw_by_recipe, bins=50)
ax.set_ylabel('Number of tweets')
ax.set_title('Distribution of number of recipe tweets')
plt.show()







## eof







