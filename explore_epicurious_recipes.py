# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:36:40 2020

Project idea for The Data Incubator: Sustainable recipe recommender

@author: sbuer
"""

# Data science libraries
import numpy as np
import pandas as pd 

# Saving and loading
import json

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


## Change some defaults for visualizing data frames
pd.set_option("max_columns", 100)
pd.set_option("max_rows", 100)


# get recipes in from csv file (omits ingredient details - dummy code)
## Source: https://www.kaggle.com/hugodarwood/epirecipes
df_rec = pd.read_csv("epicurious_recipes_2017/epi_r.csv")
# kick out duplicate recipe names
df_rec.drop_duplicates(subset="title", inplace=True)
column_labels = list(df_rec.columns)


## save column labels to csv file, then manually go through and assign a label
## to each ingredient (where possible) that fits with the estimated green house
## gas emissions for a products life cycle from "our world in data"
## Source: https://ourworldindata.org/food-choice-vs-eating-local

#df = pd.DataFrame(column_labels, columns=["epicurious_labels"])
#df.to_csv('epicurious_ingredient_sustainability.csv', index=False)

## Read translation data back in after inputting labels manually
df_link = pd.read_csv('epicurious_ingredient_sustainability.csv')


## Read in look-up table for GHG emissions / sustainability
df_GHG = pd.read_csv('GHG-emissions-by-life-cycle-stage.csv')


## Explore GHG data a bit

## Add total GHG emissions to table
df_GHG['total'] = df_GHG.loc[:,'Land use change':'Retail'].sum(axis=1)


## Visualize total GHG emissions
srtid = df_GHG['total'].sort_values().index
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,10))
ax.barh(df_GHG['Food product'][srtid], df_GHG['total'][srtid], align='center')
ax.set_yticklabels(df_GHG['Food product'][srtid])
ax.set_xlabel('kg-CO2 emissions by kg-product')
ax.set_title('Green house gas emissions by food product')
plt.show()




########
## If I have done this already, unnecessary to rerun!

## Add GHG emission data to recipe table (this can be sped up later)
GHG_em = list() # Track GHG emission over all ingredients
Ning_em = list() # Track number of ingredients per dish with emission value
for irec, recipe in enumerate(df_rec['title']):
 	if irec % 100 == 0:
		 print("Recipe {}. {}".format(irec, recipe))
 	
 	## Get ingredients for this recipe and add up GHG emissions
 	ing_idx = df_rec.iloc[irec,0:680].values == 1
 	sust_labels = df_link['sustainability_labels'][ing_idx].str.lower().dropna()
 	
 	## For simplicity drop duplicates (e.g. ham and pork will lead to
 	## two Pig Meat labels, but these labels might be redundant and we are not 
 	## concerned about quantities here anyway yet.)
 	sust_labels = sust_labels.unique()
 	
 	## Keep track of number of ingredients with GHG emission value
 	Ning_em.append(sust_labels.shape[0])
 	
 	## Add up total GHG emissions for dish
 	if len(sust_labels) > 0:
		 ghg_idx = df_GHG['Food product'].str.lower().isin(sust_labels)
		 GHG_em.append(df_GHG['total'][ghg_idx].sum())
 	else:
		 GHG_em.append(None)
		 
	## ID of dish with 0 GHG emissions (allegedly) 9739
		
		
## Add GHG emissions, number of ingredients with emissions and number of total
## ingredients to recipe dataframe
df_rec['GHG emissions'] = GHG_em
df_rec['Num_ingr_w_GHG_data'] = Ning_em
df_rec['Number of ingredients'] = df_rec.loc[:,'almond':'turkey'].sum(axis = 1)

## Save data as csv
df_rec.to_csv('epi_r_w_sust.csv', index=False)
########




## Load data (back) in
df_rec = pd.read_csv('epi_r_w_sust.csv')



## Visualize Carbon footprint of dishes (absolute)
sns.distplot(df_rec['GHG emissions'].dropna())
plt.suptitle('Estimated GHG emission distribution from 20000 recipes');


## There appear to be three clusters of Low, intermediate and high GHG output.
## Does this have to do with the number of overall ingredients? 
sns.distplot(df_rec['Number of ingredients'].dropna())

## Clearly not! Another way of looking at this:
df = df_rec.loc[:,['GHG emissions', 'Number of ingredients']].dropna()
sns.pairplot(df);
plt.suptitle('Relationship between total GHG emissions and # of ingredients');

## Surprisingly there does not seem to be much of a relationship between 
## number of ingredients and total GHG emissions, to quantify let's also do
## some statistics
import scipy
rho = scipy.stats.spearmanr(df['Number of ingredients'], df['GHG emissions'])
rho

# So statistically there definitely is a relationship, but its effect size is
# surprisingly small (Rho = 0.095)


## This relationship should be more pronounced when I consider the number
## of ingredients with an emission value
df = df_rec.loc[:,['GHG emissions', 'Num_ingr_w_GHG_data']].dropna()
sns.pairplot(df);
plt.suptitle('GHG emissions vs # of ingredients with GHG data');
rho2 = scipy.stats.spearmanr(df['Num_ingr_w_GHG_data'], df['GHG emissions'])
rho2

## As expected, this gives a much larger correlation (rho=0.48)
## Hence, there is definitely a strong incentive to make sure that most 
## ingredients can be assigned a GHG emission value, otherwise results will be
## biased. 




########
## What meals are particularly sustainable? Select the three peaks in the 
## GHG emissions data by hand:
sns.distplot(df_rec['GHG emissions'].dropna())
 
# create group vector (0 = low, 1 = medium, 2 = high)
GHG_group = np.zeros(df_rec.shape[0])
GHG_group[df_rec['GHG emissions'].between(19.001,39)] = 1
GHG_group[df_rec['GHG emissions'].between(39.001,df_rec['GHG emissions'].max())] = 2
plt.plot(GHG_group[::100])
df_rec['GHG group'] = GHG_group 


## Show 10 recipes for each subgroup
print(df_rec['title'][df_rec['GHG group'] == 0][0:10])
print(df_rec['title'][df_rec['GHG group'] == 1][0:10])
print(df_rec['title'][df_rec['GHG group'] == 2][0:10])


## Given the crazy carbon footprint of beef, how many of the dishes in the "high"
## emission group contain beef?
beef_idx = df_link['sustainability_labels'] == 'Beef (beef herd)'
df_rec['contains beef'] = df_rec.iloc[:,0:(len(beef_idx))].loc[:,beef_idx.values].any(axis=1)

sum(df_rec['contains beef'] & (df_rec['GHG group'] == 2)) / sum(df_rec['GHG group'] == 2)

## As expected almost all dishes with high carbon footprint contain beef




########
## What are the best and worst 10 recipes for GHG emissions?
GHG_sort_idx = df_rec.dropna().sort_values(by='GHG emissions').index
sust_rec = df_rec.iloc[GHG_sort_idx[0:10],:]
unsust_rec = df_rec.iloc[GHG_sort_idx[-10:],:]




#########
## Build a recommender system that suggests similar food choices, but more
## sustainable! 

## For this I want to find some users of epicurious recipes using twitter.
## I will simply assume that if somebody tweets a recipe, they also like it
## for now. 


## Twitter 
import tweepy as tw # recevent tweets (last 7 days)
import GetOldTweets3 as got # archived tweets

## Make python wait for twitter server 
import time




## Searching for recipe names and "epicurious" does not work super well. Most
## tweets are plugs from epicurious themselves or other cooking websites / 
## accounts, not home cooks / users. Instead, try to find homecooking with
## epicurious hashtags and try to assign recipes. It would also be great to be 
## able to assign a rating to the dish if possible. 
search_words = '#epicurious OR #homecooking AND recipe'
search_words = 'https://www.epicurious.com/recipes/food/views/blueberry-lemon-corn-muffins-15255'
max_tweets = 20
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(search_words)\
	.setSince("2010-01-01")\
	.setUntil("2020-05-05")\
	.setMaxTweets(max_tweets)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
## if tweets is not empty, add users with title to dictionary
if tweets:
	print(i, title, ': Tweet found!!!')
	for j, tweet in enumerate(tweets):
		print('')
		print(tweet.username)
		print(tweet.text)	



# NLP imports
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# NLP and sentiment analysis
from textblob import TextBlob
from wordcloud import WordCloud		


# Twitter API Authentication
df_twitter_access = pd.read_csv('twitter_access.csv')
ACCESS_TOKEN = df_twitter_access.ACCESS_TOKEN
ACCESS_SECRET = df_twitter_access.ACCESS_SECRET
CONSUMER_KEY = df_twitter_access.CONSUMER_KEY
CONSUMER_SECRET = df_twitter_access.CONSUMER_SECRET

# Get API access
def connect_to_twitter_OAuth():
    auth = tw.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tw.API(auth)
    return api

api = connect_to_twitter_OAuth()


	
			



## Search for recipe names and 'epicurious'
## Runs into rate limit fairly quickly (after around 400 recipes)
## To simplify only look for recipes with a rating > 4 and take periodic breaks
user_dict = {}
cnt = 0
for i, title in enumerate(df_rec['title'][0:]):
	cnt = cnt + 1
	## take a 5 minutes break from spamming the twitter server every 300 titles
	if cnt > 500:
		time.sleep(900)
		cnt = 0
	if i % 100 == 0:
		print(i, title)
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
		for j, tweet in enumerate(tweets):
			user = tweet.username
			if user in user_dict:
				user_dict[user].add(title)
			else:
				user_dict[user] = set([title])
				


## Using my make-shift user dictionary I can now create a matrix of users x 
## recipes and use it for my recommender
user_keys = user_dict.keys()
titles = list(df_rec['title'].values)
usermat = np.zeros((len(user_keys), len(titles)))
df_user_matrix = pd.DataFrame(usermat, columns = titles, index=user_keys)
for i, user in enumerate(user_keys):
	match_ids = [j for j, val in enumerate(titles) if val in user_dict[user]] 
	df_user_matrix.iloc[i, match_ids] = 1
	
print('Number of users after around 5000 recipes: ', len(user_keys))

## Save data as csv
df_user_matrix.to_csv('epi_user_matrix.csv')




## reload matrix data and collapse to two columns (very inefficient, but I can
## later delete this anyway...)
df = pd.read_csv('epi_users.csv', index_col=0)
user_list = list()
title_list= list()
for user in list(df.index):
	user_titles = list(df.loc[user,df.loc[user,:]==1].index)
	for title in user_titles:
		user_list.append(user)
		title_list.append(title)
		
df_users = pd.DataFrame({'user':user_list, 'title':title_list})




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
df_users.to_csv('')
	
print('Number of users after around 5000 recipes: ', len(user_keys))
df_users.to_csv('epi_users.csv')




## plot distribution of number of recipes per user
plt.hist(df_users.sum(axis=1).values, bins=len(user_keys))



## plot recipes most users tweeted about
recipe_tweetnum = df_users.sum(axis=0).values
srtid = np.flip(np.argsort(recipe_tweetnum).astype(int))
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,10))
ax.barh(np.asarray(titles)[srtid][0:25], recipe_tweetnum[srtid][0:25], align='center')
ax.set_xlabel('Number of twitter users')
ax.set_title('What recipe titles are tweeted about by most users?')
plt.show()



## What recipe is called "Love "?
df_rec.iloc[titles.index('Love '),:]


## It might be worth to kick out common single word recipe names from the
## twitter search!




########
## Future directions:
##
## Parentheses indicate optional or later steps
##
## (i) Improve twitter user search
##         > Avoid single common word recipe names
##         > Try looking for urls of recipes specifically
##        (> Maybe even try to assign sentiment to tweets)
## 
## (ii) Build recommender system
##  	   > User-based collabroative filtering
## 	       > Item-based collaborative filtering
## 	      (> Once the collab. filt. works, try neural networks)
##
## (iii) More and better data
## 	       > Get more recipe data 
## 	       > Improve on dummy coding - include quantities of ingredients
## 	       > Improve on GHG emissions labels for ingredients (possible to find
## 	         additional, complementary sources?)


## Our first user, Susanna43b has lots of recipe tweets. I can use her as an
## example, leaving out half her tweets for model evaluation
df_users = pd.read_csv('epi_users.csv', index_col=0)



## I will start with a simple K-nearest-neighbor predictor, i.e. we will 
## recommend to Susanna43b recipes that users tweeted who are similar to her
## (her neighbors in terms of where they lie in the multidimensional recipe
## space), but which Susanna43b has not tweeted about herself. We can then 
## evaluate the performance of our prediction by seeing how many left-out 
## tweets are suggested. 



## ML tools using sklearn
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# For validation, make sure each recipe has at least 2 users tweeting about it
multi_tweet_recipes = df_users['title'].unique()[df_users['title'].value_counts() > 1]
df_multi_tweets = df_users[df_users['title'].isin(multi_tweet_recipes)]

# and each user has tweeted about at least 2 recipes
multi_tweet_users = df_multi_tweets['user'].unique()[df_multi_tweets['user'].value_counts() > 1]
df_multi_tweets = df_multi_tweets[df_multi_tweets['user'].isin(multi_tweet_users)]

# Create sparse matrix of users x recipes with "ratings" as cells, i.e. for
# now this is just whether the recipe was tweeted or not. Also:
# Create training and test sets (split equally, when uneven split to favour
## training set).
from scipy.sparse import csr_matrix
import random as rnd

users = df_multi_tweets['user'].unique()
titles = df_multi_tweets['title'].unique()
Nx = users.shape[0]
Ny = titles.shape[0]
user_rec_mat = csr_matrix((Nx, Ny), dtype=np.int8)
train_mat = csr_matrix((Nx, Ny), dtype=np.int8)
test_mat = csr_matrix((Nx, Ny), dtype=np.int8)
for u, user in enumerate(users):
	user_titles = list(df_users['title'][df_users['user'] == user])
	match_ids = [j for j, val in enumerate(titles) if val in user_titles] 
	user_rec_mat[u, match_ids] = 1
	trn_ids = rnd.sample(match_ids,np.ceil(len(match_ids)/2).astype(int))
	tst_ids = np.setdiff1d(match_ids, trn_ids)
	train_mat[u, trn_ids] = 1
	test_mat[u, tst_ids] = 1

user_rec_mat
plt.imshow(user_rec_mat.toarray())
plt.colorbar()
plt.show()



########
## Train model
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
from skopt import forest_minimize


## From the lightfm documentation:
## https://making.lyst.com/lightfm/docs/home.html

## Create data object in lightfm format (based on example movielens data)
# from lightfm.datasets import fetch_movielens
# data = fetch_movielens(min_rating=5.0)
# plt.imshow(data['item_features'].toarray())
# data_dict = {'train':train_mat, 
# 			 'test':test_mat, 
# 			 'item_features': ,
# 			 'item_feature_labels': ,
# 			 'item_labels':}


## Create a model instance with the desired latent dimensionality:
model = LightFM(no_components=30)

## Assuming train is a (no_users, no_items) sparse matrix (with 1s denoting 
## positive, and -1s negative interactions), you can fit a traditional matrix factorization model by calling:
model.fit(train_mat, epochs=20)


print("Train precision: %.2f" % precision_at_k(model, train_mat, k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, test_mat, k=5).mean())

## This will train a traditional MF model, as no user or item features have been supplied.
## To get predictions, call model.predict:
predictions = model.predict(test_user_ids, test_item_ids)



	

model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)

model = model.fit(user_rec_mat,
                  epochs=100,
                  num_threads=16, verbose=False)







def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 5, show = True):
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items), item_features=books_metadata_csr))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print ("User: " + str(user_id))
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
			

sample_recommendation_user(model, user_book_interaction, 'ff52b7331f2ccab0582678644fed9d85', user_dict, item_dict)



	
	

# Features (independent variables, in our case users)
X = df_users['user'].values
X[0:5]

# Labels (dependent vairable, in our case recipe titles)
y = df_users['title'].values
y[0:5]


# divide data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify = y)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

# this last bit is just to check whether I get the same answer with the standardscaler method (see below)
X_test_z_manual = (X_test - X_train.mean()) / np.std(X_train)
X_test_z_manual[0:5]


cnts_train = pd.value_counts(y_train)
cnts_test = pd.value_counts(y_test)
print(cnts_train.PAIDOFF/cnts_train.COLLECTION)
print(cnts_test.PAIDOFF/cnts_test.COLLECTION)


Xz_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
Xz_test = preprocessing.StandardScaler().fit(X_train).transform(X_test)
Xz_test[0:5]


## iterate through 20 different Ks to find the best K
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(Xz_train,y_train)
    yhat=neigh.predict(Xz_test)
    mean_acc[n-1] = accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


## Visualize decoding accuracies
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

# refit model for best k
neigh = KNeighborsClassifier(n_neighbors = mean_acc.argmax()+1).fit(Xz_train,y_train)









## ML tools using surprise
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans


## Put data into surprise format
reader = Reader(rating_scale=(1))

# Loads Pandas dataframe
data = Dataset(df_users)



























## eof