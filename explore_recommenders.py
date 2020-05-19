#!/usr/bin/env python
# coding: utf-8

# # Project idea for The Data Incubator: Sustainable recipe recommender
# 
# Created on Sun May  3 18:36:40 2020
# 
# @author: sbuergers@gmail.com

# The majority of people on this planet are concerned with its future and the possibly catastrophic effects of global warming in both humanitarian and economic terms. While many people believe that eating food that is produced locally lowers their carbon footprint, research have shown that green house gas emissions due to food transport have very little overall impact.  
# 
# To assist people in shopping sustainably while at the same time cooking delicious food, I want to build a recipe recommendation engine that leverages publicly available ingredient lists for many recipes, and scientific findings on associated green house gas emissions. The goal is to suggest alternative recipes that are in line with users' tastes, but have a reduced carbon footprint to make for an easy change of cooking habits.  
# 
# As a proof of concept I have used a ~20000 recipe dataset from Kaggle and used the GetOldTweets3 API to simulate Users and their recipe preferences simply by checking if they tweeted about them. The majority of recipes could be assigned green house gas emission values and after searching for <4000 recipes I could obtain 207 user accounts from twitter after a simple first search. 
#   
# Hence, I believe that this project is feasible and more importantly interesting for a large group of people who care to change their cooking behaviour to be more sustainable without sacrificing food enjoyment. 

# ### Set up environment

# In[1]:


# Data science libraries
import numpy as np
import pandas as pd 

# Handle sparse matrices efficiently
from scipy.sparse import csr_matrix

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

## Change some defaults for visualizing data frames
pd.set_option("max_columns", 15)
pd.set_option("max_rows", 15)


# ### Load data 
# (I downloaded it from kaggle manually)

# In[2]:

#cd D:\data science\nutrition


# In[3]:


# get recipes from csv file 
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


# ### Explore green house gas (GHG) emissions data
# for more details see https://ourworldindata.org/food-choice-vs-eating-local

# In[4]:


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


## Visualize total GHG emissions of only some products
srtid = df_GHG['total'].sort_values().index
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5,5))
ax.barh(df_GHG['Food product'][srtid][10::2], df_GHG['total'][srtid][10::2], align='center')
ax.set_yticklabels(df_GHG['Food product'][srtid][10::2])
ax.set_xlabel('kg-CO2 emissions by kg-product')
ax.set_title('Green house gas emissions by food product')
plt.show()


# ### (re-)load recipe data with GHG emission column

# In[5]:


df_rec = pd.read_csv('epi_r_w_sust.csv')


# ### Visualize Carbon footprint of dishes (absolute)

# In[6]:


sns.distplot(df_rec['GHG emissions'].dropna())
plt.suptitle('Estimated GHG emission distribution from 20000 recipes');


# ### Explore relationship between number of ingredients and GHG emissions

# In[7]:


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


# ### Example recipes with different degrees of sustainability

# In[8]:


# create group vector (0 = low, 1 = medium, 2 = high)
GHG_group = np.zeros(df_rec.shape[0])
GHG_group[df_rec['GHG emissions'].between(19.001,39)] = 1
GHG_group[df_rec['GHG emissions'].between(39.001,df_rec['GHG emissions'].max())] = 2
df_rec['GHG group'] = GHG_group 


## Show 10 recipes for each subgroup
print(' ')
print('Sustainable:')
print(df_rec['title'][df_rec['GHG group'] == 0][0:10])
print(' ')
print('Not sustainable:')
print(df_rec['title'][df_rec['GHG group'] == 1][0:10])
print(' ')
print('Absolutely not sustainable:')
print(df_rec['title'][df_rec['GHG group'] == 2][0:10])


# ### Do all recipes with high green house gas emissions contain beef?
# No, but more than 90 percent do.

# In[9]:


beef_idx = df_link['sustainability_labels'] == 'Beef (beef herd)'
df_rec['contains beef'] = df_rec.iloc[:,0:(len(beef_idx))].loc[:,beef_idx.values].any(axis=1)

sum(df_rec['contains beef'] & (df_rec['GHG group'] == 2)) / sum(df_rec['GHG group'] == 2)


# ### What are the best and worst 10 recipes for GHG emissions?

# In[10]:


GHG_sort_idx = df_rec.dropna().sort_values(by='GHG emissions').index
sust_rec = df_rec.iloc[GHG_sort_idx[0:10],:]
unsust_rec = df_rec.iloc[GHG_sort_idx[-10:],:]
print(' ')
print('Most sustainable recipes:')
print(sust_rec['title'])
print(' ')
print('Least sustainable recipes:')
print(unsust_rec['title'])


# ## Build a recommender system that suggests similar food choices, but more sustainable! 
# 
# For this I want to find some users of epicurious recipes using twitter.
# I will simply assume that if somebody tweets a recipe, they also like it.

# __Load user-recipe table__ created from tweets

# In[11]:


df_users = pd.read_csv('epi_users.csv', index_col=0)


# In[12]:


df_users.head()


# __Create sparse matrix of users x recipes__ with "ratings" as cells, i.e. for
# now this is just whether the recipe was tweeted or not. 
# 

# In[13]:


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




# In[ ]

# Show user x title matrix for example recipes in TDI presentation

sample_rec_list = ['Mixed-Berry Chiffon Cake with Almond Cream Cheese Frosting ',
				   'Cherry-Chocolate Shortcakes with Kirsch Whipped Cream ',
				   'Spice Cake with Blackberry Filling and Cream Cheese Frosting ',
				   'Cherry-Almond Clafouti ' ]
titles.tolist().index(sample_rec_list[0])
titles.tolist().index(sample_rec_list[1])
titles.tolist().index(sample_rec_list[2])
titles.tolist().index(sample_rec_list[3])

# none of these is in the list... so let's just fake it and use random recipes



user_rec_mat[]

sample_idx = df_rec.index[df_rec['title'].isin(sample_rec_list)].values

df = df_rec.loc[sample_idx,'#cakeweek':'turkey']
item_sim_sample = cosine_similarity(df)

sns.heatmap(item_sim_sample, annot=True, cmap='viridis')



# Plot distribution of number of recipes per user

# ### Distribution of number of recipe tweets by users

# In[14]:


## Compute number of recipe tweets by user
N_tw_by_user = np.sum(user_rec_mat.toarray(), axis=1)

## plot distribution of number of recipe tweets per user
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(N_tw_by_user, bins=50)
ax.set_ylabel('Number of recipe tweets')
ax.set_title('Distribution of number of recipe tweets by users')
plt.show()


# ### Distribution of number of recipe tweets

# In[15]:


## Compute number of tweets by recipe
N_tw_by_recipe = np.sum(user_rec_mat.toarray(), axis=0)

## plot distribution of number of tweets per recipe
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(N_tw_by_recipe, bins=50)
ax.set_ylabel('Number of tweets')
ax.set_title('Distribution of number of recipe tweets')
plt.show()


# ### What recipe titles are tweeted about by most users?

# In[16]:


users = df_users['user'].unique()
titles = df_users['title'].unique()
recipe_tweetnum = np.sum(user_rec_mat.toarray(), axis=0)
srtid = np.flip(np.argsort(recipe_tweetnum).astype(int))
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,10))
ax.barh(np.asarray(titles)[srtid][0:45], recipe_tweetnum[srtid][0:45], align='center')
ax.set_xlabel('Number of twitter accounts')
ax.set_title('What recipe titles are tweeted about by most users?')
plt.show()


# # Build recommender system (content based)
# 
# For this we need the ingredients of each dish to quantify how similar different dishes are. We can then use that similarity to find dishes that are in line with what users have previously tweeted about.

# In[17]:


df_rec = pd.read_csv('epi_r_w_sust.csv')
df_rec.head()


# For starters let's consider all ingredient columns from the original epicurious dataset (as baseline). Of note: Recipes with more ingredients will automatically be more different, but a dish with 5 ingredients that all overlap with one with 15 ingredients might be more similar than one with 10 ingredients having 5 overlapping ones with the 15 ingredient recipe. For now I will ignore this. Also note that not all columns actually capture ingredients. Many capture other aspects such as 'bon appetit', or '3 ingredient recipe'...

# In[18]:


col_labels = df_rec.columns.tolist()


# In[19]:


for i, lbl in enumerate(col_labels):
    print(i, lbl)


# What similarity measure should I choose? Cosine similarity disregards magnitudes and focuses on orientation. However, it is sensitive to the mean. Hence I should center before using it. Other measures of similarity like euclidean or mahalanobis distance are sensitive to the magnitude and require standardizing to put each variable/ dimension in the same scale. 

# __cosine similarity__

# In[20]:


from sklearn.metrics.pairwise import cosine_similarity

df = df_rec.loc[:,'#cakeweek':'turkey']
#df = df.subtract(df.mean())

item_sim = cosine_similarity(df)


# In[21]:


np.shape(item_sim)


# In[24]:

# show items for presentation:
sample_rec_list = ['Mixed-Berry Chiffon Cake with Almond Cream Cheese Frosting ',
				   'Cherry-Chocolate Shortcakes with Kirsch Whipped Cream ',
				   'Spice Cake with Blackberry Filling and Cream Cheese Frosting ',
				   'Cherry-Almond Clafouti ' ]
sample_idx = df_rec.index[df_rec['title'].isin(sample_rec_list)].values

df = df_rec.loc[sample_idx,'#cakeweek':'turkey']
item_sim_sample = cosine_similarity(df)

sns.heatmap(item_sim_sample, annot=True, cmap='viridis')



# For convenience write a __function showing the closest N recipes__ in df_rec for a given input recipe name given the similarity matrix SM

# In[25]:


def find_related_recipes(name, df_rec, N, SM):
    rec_id = df_rec['title'] == name
    similarities = np.flip(np.sort(SM[rec_id,:],axis=1)[:,-N:])[0]
    rel_rec_ids = np.flip(np.argsort(SM[rec_id,:],axis=1)[:,-N:])[0]
    related_recipes = df_rec.iloc[rel_rec_ids,:]
    return(related_recipes, similarities)

	
def find_related_sustainable_recipes(name, df_rec, N, SM):
	rec_id = df_rec['title'] == name
	similarities = np.flip(np.sort(SM[rec_id,:],axis=1)[:,-N:])[0]
	rel_rec_ids = np.flip(np.argsort(SM[rec_id,:],axis=1)[:,-N:])[0]
	ghg_em = df_rec.iloc[rel_rec_ids,:]['GHG emissions']
	# combine similarities and sustainabilities
	# compute ghg emissions of each recipe relative to the searched recipe
	ghg_em_norm = ghg_em / df_rec[df_rec['title']==name]['GHG emissions'].values
	comb_score = ghg_em_norm + (1-similarities)
	rel_rec_ids = rel_rec_ids[np.argsort(comb_score)]
	related_recipes = df_rec.iloc[rel_rec_ids,:]
	return(related_recipes, similarities)


# Helper __function to print out recipe information__ including ingredients and tags

# In[26]:


def show_recipe_ingredients(name, df_rec):
    df_recipe = df_rec.loc[df_rec['title'] == name, :]
    df_recipe = df_recipe.loc[:, (df_recipe !=0).any(axis=0)]
    N = len(df_recipe.columns)
    print('----------------------------------------------------------------')
    print('Name: ', df_recipe['title'].values)
    print('................................................................')
    print('Rating: ', df_recipe['rating'].values)
    print('Calories: ', df_recipe['calories'].values)
    print('Protein: ', df_recipe['protein'].values)
    print('Fat: ', df_recipe['fat'].values)
    print('Sodium: ', df_recipe['sodium'].values)
    print('Estimated GHG emissions: ', df_recipe['GHG emissions'].values)
    print('Ingredients with GHG estimate: ', df_recipe['Num_ingr_w_GHG_data'].values)
    print('Number of ingredients or tags: ', df_recipe['Number of ingredients'].values)
    print('Tags and ingredients: ')
    for i, tag in enumerate(df_recipe.columns.tolist()[6:(N-3)]):
        print(i, tag)
    print(' ')


# __Find related recipes to a random recipe__ and print ingredient and tag information 

# In[27]:


import random

recipe = random.choice(df_rec['title'])
print(recipe)
N_rel_rec = 7
rel_rec, sim = find_related_recipes(recipe, df_rec, N_rel_rec, item_sim)


# Show output (input recipe, and respective output recipes)

# In[28]:

print('FIND ALTERNATIVES:')
print('MAIN RECIPE:')
show_recipe_ingredients(recipe, df_rec)
## skip first, because it again shows the recipe itself
for i, sim_recipe in enumerate(rel_rec['title'][1:]):
    print(' ')
    print('SIMILAR RECIPES:', i+1)
    print('Cosine similarity:', sim[i+1])
    show_recipe_ingredients(sim_recipe, df_rec)


print('FIND SUSTAINABLE ALTERNATIVES:')
print('MAIN RECIPE:')
show_recipe_ingredients(recipe, df_rec)
## skip first, because it again shows the recipe itself
for i, sim_recipe in enumerate(rel_rec['title'][1:]):
    print(' ')
    print('SIMILAR RECIPES:', i)
    print('Cosine similarity:', sim[i])
    show_recipe_ingredients(sim_recipe, df_rec)

# ## Collaborative filtering recommender
# 
# In collaborative filtering we assign recommendations based on other users that are similar to us. For example, when James like recipes A and B and Jane like recipes A, B and C, we might consider recommending recipe C to James. 
# 
# From https://realpython.com/build-recommendation-engine-collaborative-filtering/:
# 
# User-based: For a user U, with a set of similar users determined based on rating vectors consisting of given item ratings, the rating for an item I, which hasn’t been rated, is found by picking out N users from the similarity list who have rated the item I and calculating the rating based on these N ratings.
# 
# Item-based: For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, the rating by a user U, who hasn’t rated it, is found by picking out N items from the similarity list that have been rated by U and calculating the rating based on these N ratings.

# In[29]:


plt.imshow(user_rec_mat.toarray())
plt.colorbar()
plt.show()


# It is clear that the matrix is very sparse. And a few accounts tweeted many recipes, whilst a few recipes were tweeted by many users. 

# __Compute user-user similairty matrix__, remember that user_rec_mat is users x recipes

# In[30]:


user_rec_mat.toarray()[0:10,0:10]


# In[31]:


user_sim = cosine_similarity(user_rec_mat)
user_sim.shape


# In[32]:


plt.imshow(user_sim)
plt.colorbar()
plt.show()


# It is still very evident that the data is extremely sparse - several users have exactly the same recipe tweets!

# Define a __function that makes recommendations__ based on either similar users (type='user') or items (type='item')
# 
# adapted from https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

# In[37]:


# similarity = user_sim
# ratings = user_rec_mat


# # In[ ]:


# # pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T


# # In[34]:


# plt.imshow(ratings_diff)
# plt.colorbar()
# plt.show()


# # In[35]:


# pred.shape


# # In[36]:


# plt.imshow(pred[0:20, 0:20])
# plt.colorbar()
# plt.show()


# # It makes sense to __subtract a user's average rating before comparing with other users__, because some users might generally be more liberal or conservative, and a rating of 3 for user A might be just as good as a rating of 5 for user B. Of course, as long as I only use binary ratings - either tweeting or not tweeting, this is irrelevant. But it might become relevant eventually (for example I could do sentiment analysis on the tweets). 
# # 
# # To do this properly I have to replace all 0s with NaNs, because otherwise 0 will be seen as an implicit rating, which does not seem fair in this case. 
# # 

# # In[ ]:


# def predict(ratings, similarity, type='user'):
#     if type == 'user':
#         mean_user_rating = np.true_divide(ratings.sum(1),(ratings!=0).sum(1))
#         #We use np.newaxis so that mean_user_rating has same format as ratings
#         ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
#         pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
#     elif type == 'item':
#         pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
#     return pred


# # In[ ]:




# In[ ]:
# I can use the surprise module to fit a K-nearest-neighbor algorithm to the 
# user-recipe matrix, finding the K nearest users for each user and predicting
# items the user might like based on what his/her neighbors have liked, but
# which they haven't tried yet. This is known as user based collaborative
# filtering. 

# When you have a lot of users it can also make sense to do item-based 
# collaborative filtering, but we will not focus on that now. 

# First, prepare the data to be in surprise format:
# We want the first column of a dataframe to contain recipes (item id), the
# second column users (user id), and the third column the rating.

from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import SVD
from surprise.model_selection import train_test_split

from collections import defaultdict


# add rating column to df_users dataframe
df_users['rating'] = np.ones(df_users.shape[0])
df = df_users
df.columns = ['user', 'item', 'rating']

# assume unrated items are neutral, and rated items perfect scores
reader = Reader(rating_scale=(-1, 1))

# put data into surprise format
data = Dataset.load_from_df(df, reader)



# In[36]:

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": True,  # Compute similarities between user or items
}
# KNNWithMeans takes into account mean rating of each user (unnecessary for 
# now, as all ratings are 1)
#algo = KNNWithMeans(k=10, min_k=1, sim_options=sim_options)
algo = KNNBasic(k=10, min_k=1, sim_options=sim_options)



# In[ ]:

# from https://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



# In[33]:
	
# First train a KNN algorithm on training set.
# algo = SVD()
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=5)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
	
	
	

# Divide into training and test sets (there is also full cross validation)
# create training set
trainset, testset = train_test_split(data, test_size=.5)

# fit model
algo.fit(trainset).test(testset)



# Use the full training set to make predictions!
# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo.fit(trainset)

# test algorithm on testset
predictions = algo.test(testset)

# Retrieve top N predictions for each item in predictions (in the test set)


# make prediction
recipe = random.choice(df_users['item'].unique())

# pick a recipe that has multiple users rating it
uid = 613
iid = 2
print(users[uid], ':', titles[iid])
prediction = algo.predict(users[uid], titles[iid], 1)
prediction.est


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# __Divide data into training and test sets__. For this, make sure to only include recipes with at least two ratings such that both training and test set have at least one. 

# In[38]:


# For validation, make sure each recipe has at least 2 users tweeting about it
multi_tweet_recipes = df_users['title'].unique()[df_users['title'].value_counts() > 1]
df_multi_tweets = df_users[df_users['title'].isin(multi_tweet_recipes)]

# and each user has tweeted about at least 2 recipes
multi_tweet_users = df_multi_tweets['user'].unique()[df_multi_tweets['user'].value_counts() > 1]
df_multi_tweets = df_multi_tweets[df_multi_tweets['user'].isin(multi_tweet_users)]


# Create sparse matrix of users x recipes with "ratings" as cells, i.e. for
# now this is just whether the recipe was tweeted or not. Also:
# Create training and test sets (split equally, when uneven split to favour
# training set).

# In[49]:


df_multi_tweets.shape


# In[48]:


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
    trn_ids = random.sample(match_ids,np.ceil(len(match_ids)/2).astype(int))
    tst_ids = np.setdiff1d(match_ids, trn_ids)
    train_mat[u, trn_ids] = 1
    test_mat[u, tst_ids] = 1

user_rec_mat
plt.imshow(user_rec_mat.toarray())
plt.colorbar()
plt.show()


# __Get LightFM package__, which was build to facilitate building recommender systems.
# 
# The following code was adapted from:
# * https://making.lyst.com/lightfm/docs/quickstart.html

# In[36]:


from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
from skopt import forest_minimize


# Build __a simple model using matrix factorization__ (item-based collaborative filtering)

# In[46]:


## Create a model instance with the desired latent dimensionality:
model = LightFM(no_components=30)

## Assuming train is a (no_users, no_items) sparse matrix (with 1s denoting 
## positive, and -1s negative interactions), you can fit a traditional matrix factorization model by calling:
model.fit(train_mat, epochs=2000)


# In[51]:


print("Train precision: %.2f" % precision_at_k(model, train_mat, k=3).mean())
print("Test precision: %.2f" % precision_at_k(model, test_mat, k=3).mean())


# In[ ]:


data_dict = {'train':train_mat, 
            'test':test_mat, 
            'item_labels':}


# In[ ]:


def sample_recommendation(model, data, user_ids):

    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


# In[ ]:


sample_recommendation(model, data, [3, 25, 450])


# In[ ]:





# In[ ]:





# In[ ]:


## From https://medium.com/@wwwbbb8510/python-implementation-of-baseline-item-based-collaborative-filtering-2ba7c8960590
##
## by 

# function of building the item-to-item weight matrix
def build_w_matrix(adjusted_ratings, load_existing_w_matrix):
   # define weight matrix
   w_matrix_columns = ['movie_1', 'movie_2', 'weight']
   w_matrix=pd.DataFrame(columns=w_matrix_columns)

   # load weight matrix from pickle file
   if load_existing_w_matrix:
       with open(DEFAULT_PARTICLE_PATH, 'rb') as input:
           w_matrix = pickle.load(input)
       input.close()

   # calculate the similarity values
   else:
       distinct_movies = np.unique(adjusted_ratings['movieId'])

       i = 0
       # for each movie_1 in all movies
       for movie_1 in distinct_movies:

           if i%10==0:
               print(i , "out of ", len(distinct_movies))

           # extract all users who rated movie_1
           user_data = adjusted_ratings[adjusted_ratings['movieId'] == movie_1]
           distinct_users = np.unique(user_data['userId'])

           # record the ratings for users who rated both movie_1 and movie_2
           record_row_columns = ['userId', 'movie_1', 'movie_2', 'rating_adjusted_1', 'rating_adjusted_2']
           record_movie_1_2 = pd.DataFrame(columns=record_row_columns)
           # for each customer C who rated movie_1
           for c_userid in distinct_users:
               print('build weight matrix for customer %d, movie_1 %d' % (c_userid, movie_1))
               # the customer's rating for movie_1
               c_movie_1_rating = user_data[user_data['userId'] == c_userid]['rating_adjusted'].iloc[0]
               # extract movies rated by the customer excluding movie_1
               c_user_data = adjusted_ratings[(adjusted_ratings['userId'] == c_userid) & (adjusted_ratings['movieId'] != movie_1)]
               c_distinct_movies = np.unique(c_user_data['movieId'])

               # for each movie rated by customer C as movie=2
               for movie_2 in c_distinct_movies:
                   # the customer's rating for movie_2
                   c_movie_2_rating = c_user_data[c_user_data['movieId'] == movie_2]['rating_adjusted'].iloc[0]
                   record_row = pd.Series([c_userid, movie_1, movie_2, c_movie_1_rating, c_movie_2_rating], index=record_row_columns)
                   record_movie_1_2 = record_movie_1_2.append(record_row, ignore_index=True)

           # calculate the similarity values between movie_1 and the above recorded movies
           distinct_movie_2 = np.unique(record_movie_1_2['movie_2'])
           # for each movie 2
           for movie_2 in distinct_movie_2:
               print('calculate weight movie_1 %d, movie_2 %d' % (movie_1, movie_2))
               paired_movie_1_2 = record_movie_1_2[record_movie_1_2['movie_2'] == movie_2]
               sim_value_numerator = (paired_movie_1_2['rating_adjusted_1'] * paired_movie_1_2['rating_adjusted_2']).sum()
               sim_value_denominator = np.sqrt(np.square(paired_movie_1_2['rating_adjusted_1']).sum()) * np.sqrt(np.square(paired_movie_1_2['rating_adjusted_2']).sum())
               sim_value_denominator = sim_value_denominator if sim_value_denominator != 0 else 1e-8
               sim_value = sim_value_numerator / sim_value_denominator
               w_matrix = w_matrix.append(pd.Series([movie_1, movie_2, sim_value], index=w_matrix_columns), ignore_index=True)

           i = i + 1

       # output weight matrix to pickle file
       with open(DEFAULT_PARTICLE_PATH, 'wb') as output:
           pickle.dump(w_matrix, output, pickle.HIGHEST_PROTOCOL)
       output.close()

   return w_matrix


# In[ ]:





# In[ ]:



## EOF



