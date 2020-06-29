# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:49:11 2020

Convert epi_reviews json dictionary to pandas csv file.

INPUT:
	filename of the reviews json file (default = r'epi_reviews_25plus_final_w_usernames')
	pathname of the reviews json file (default = r'D:\data science\nutrition\')
	
OUTPUT:
	Saves the converted json file as a dataframe in csv format (with the same 
															 filename)

@author: sbuer
"""

import os
import pandas as pd

def convert_reviews_dict_to_df(pathname=r'D:\data science\nutrition', 
							   filename=r'epi_reviews_75plus_w_usernames'):
	
	with open(os.path.join(pathname, filename + r".txt")) as json_file:
	    data = json.load(json_file)
	
	# create dataframe of the form 
	# idx, user, title, rating, (sentiment); leaving out sentiment for now
	recipe_titles = list(data.keys())
	user = list()
	title = list()
	rating = list()
	for irec, rec_title in enumerate(recipe_titles):
		
		# progress
		if (irec % (len(recipe_titles)/15) == 0):
			print(irec, '- Processing reviews for', rec_title)
					
		# all reviews of this recipe
		recipe = data[rec_title]
		
		# if there are reviews, go through them
		if recipe:
			
			for irev, review in enumerate(recipe):

				# when username is not "", i.e. empty, append info to lists 
				if review['username']: 
					user.append(review['username'])
					title.append(rec_title)
					rating.append(review['rating'])
					
					# TODO add column for sentiment of review
					# ...
					
	# create dataframe from lists
	df_users = pd.DataFrame({'user':user, 'title':title, 'rating':rating})				
				
	# Remove rows where rating is NaN
	df_users.dropna(inplace=True)
	
	# Some users gave multiple reviews for the same recipe. In that case average
	# over their ratings and collapse to one entry.
	df_users = df_users.groupby(['user', 'title']).mean().reset_index()
		
	# add column with centered rating by user
	rating_c = df_users.groupby('user').transform(lambda x: (x - x.mean()))
	df_users['rating_c'] = rating_c
	
	# save to csv
	df_users.to_csv(os.path.join(pathname, filename + r".csv"))



if __name__ == '__main__':
    convert_reviews_dict_to_df()


# eof