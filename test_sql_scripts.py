# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:49:27 2020

Test SQL scripts for production

@author: sbuer
"""

# for loading environment variables
import os

# needed to load environment variables from .env file
from dotenv import load_dotenv 
load_dotenv()

import psycopg2 as ps
from psycopg2 import sql
import pandas as pd


# Postgres connection class used for all interactions with the postgres AWS DB
class postgresConnection():

    def __init__(self):
        self.connect()

    def connect(self):
        '''
        DESCRIPTION:
            Create connection to AWS RDS postgres DB and cursor
        '''
        self.conn = ps.connect(
                        host=os.environ.get('AWS_POSTGRES_ADDRESS'),
                        database=os.environ.get('AWS_POSTGRES_DBNAME'),
                        user=os.environ.get('AWS_POSTGRES_USERNAME'),
                        password=os.environ.get('AWS_POSTGRES_PASSWORD'),
                        port=os.environ.get('AWS_POSTGRES_PORT'))
        self.cur = self.conn.cursor()

    def _dbsrr_query(func):
        """
        DECORATOR: Basically a try except for functions that query the postgres
            DB. When the connection fails, it tries to reconnect automatically
        """
        def func_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ps.OperationalError:
                return self.connect()
        return func_wrapper

    @_dbsrr_query
    def fuzzy_search(self, search_term, search_column="url", N=160):
        """
        DESCRIPTION:
            Searches in recipes table column url for strings that include the
            search_term. If none do, returns the top N results ordered
            by edit distance in ascending order.
        INPUT:
            cur: psycopg2 cursor object
            search_term (str): String to look for in search_column
            search_column (str): Column to search (default="url")
            N (int): Max number of results to return
        OUTPUT:
            fuzzyMatches (list): DB output (list of lists - rows x columns)
        """
        # Most similar urls by edit distance that actually contain the
        # search_term
        self.cur.execute(sql.SQL(
                    """
                    SELECT "recipesID", "title", "url", "perc_rating",
                        "perc_sustainability", "review_count", "image_url",
                        "emissions", "prop_ingredients",
                        LEVENSHTEIN({}, %s) AS "edit_dist"
                    FROM public.recipes
                    WHERE {} LIKE %s
                    ORDER BY "edit_dist" ASC
                    LIMIT %s
                    """).format(sql.Identifier(search_column),
                                sql.Identifier(search_column)),
                    [search_term, '%'+search_term+'%', N])
        fuzzyMatches = self.cur.fetchall()

        # If no results contain the search_term
        if not fuzzyMatches:
            self.cur.execute(sql.SQL(
                        """
                        SELECT "recipesID", "title", "url", "perc_rating",
                            "perc_sustainability", "review_count", "image_url",
                            "emissions", "prop_ingredients",
                            LEVENSHTEIN({}, %s) AS "edit_dist"
                        FROM public.recipes
                        ORDER BY "edit_dist" ASC
                        LIMIT %s
                        """).format(sql.Identifier(search_column)),
                        [search_term, N])
            fuzzyMatches = self.cur.fetchall()
        return fuzzyMatches

    @_dbsrr_query
    def query_content_similarity_ids(self, search_term, search_column="url"):
        """
        DESCRIPTION:
            Searches in connected postgres DB for a search_term in
            search_column and returns recipeIDs of similar recipes based
            on content similarity.
        INPUT:
            cur: psycopg2 cursor object
            search_term (str): Search term
            search_column (str): Database column name (default = "url")
        OUTPUT:
            CS_ids (tuple): Content based similarity ID vector ordered by
                similarity in descending order
        """
        self.cur.execute(sql.SQL(
                    """
                    SELECT * FROM public.content_similarity200_ids
                    WHERE "recipeID" = (
                        SELECT "recipesID" FROM public.recipes
                        WHERE {} = %s)
                    """).format(sql.Identifier(search_column)),
                    [search_term])
        CS_ids = self.cur.fetchall()[0][1::]
        CS_ids = tuple([abs(int(CSid)) for CSid in CS_ids])
        return CS_ids

    @_dbsrr_query
    def query_content_similarity(self, search_term, search_column="url"):
        """
        DESCRIPTION:
            Searches in connected postgres DB for a search_term in
            search_column and returns content based similarity.
        INPUT:
            cur: psycopg2 cursor object
            search_term (str): Search term
            search_column (str): Database column name (default = "url")
        OUTPUT:
            CS (tuple): Content based similarity vector ordered by
                similarity in descending order
        """
        self.cur.execute(sql.SQL(
                    """
                    SELECT * FROM public.content_similarity200
                    WHERE "recipeID" = (
                        SELECT "recipesID" FROM public.recipes
                        WHERE url = %s)
                    """).format(), [search_term])
        CS = self.cur.fetchall()[0][1::]
        CS = tuple([abs(float(s)) for s in CS])
        return CS

    @_dbsrr_query
    def query_similar_recipes(self, CS_ids):
        """
        DESCRIPTION:
            fetch recipe information of similar recipes based on the recipe IDs
            given by CS_ids
        INPUT:
            cur: psycopg2 cursor object
            CS_ids (tuple): Tuple of recipe IDs
        OUTPUT:
            recipes_sql (list): List of lists (row x col)
        """
        self.cur.execute(sql.SQL(
                    """
                    SELECT "recipesID", "title", "ingredients",
                        "rating", "calories", "sodium", "fat",
                        "protein", "emissions", "prop_ingredients",
                        "emissions_log10", "url", "servings", "recipe_rawid",
                        "image_url", "perc_rating", "perc_sustainability",
                        "review_count"
                    FROM public.recipes
                    WHERE "recipesID" IN %s
                    """).format(), [CS_ids])
        recipes_sql = self.cur.fetchall()
        return recipes_sql

    @_dbsrr_query
    def exact_recipe_match(self, search_term):
        '''
        DESCRIPTION:
            Return True if search_term is in recipes table of
            cur database, False otherwise.
        '''
        self.cur.execute(sql.SQL("""
                    SELECT * FROM public.recipes
                    WHERE "url" = %s
                    """).format(), [search_term])
        if self.cur.fetchall():
            return True
        else:
            return False

    def content_based_search(self, search_term):
        '''
        DESCRIPTION:
            return the 200 most similar recipes to the url defined
            in <search term> based on cosine similarity in the "categories"
            space of the epicurious dataset.
        INPUT:
            cur: psycopg2 cursor object
            search_term (str): url identifier for recipe (in recipes['url'])
        OUTPUT:
            results (dataframe): Recipe dataframe similar to recipes, but
                containing only the Nsim most similar recipes to the input.
                Also contains additional column "similarity".
        '''
        # Select recipe IDs of 200 most similar recipes to reference
        CS_ids = self.query_content_similarity_ids(search_term)

        # Also select the actual similarity scores
        CS = self.query_content_similarity(search_term)

        # Finally, select similar recipes themselves
        # Get only those columns I actually use to speed things up
        # Note that column names are actually different in sql and pandas
        # So if you want to adjust this, adjust both!
        # TODO: Make column names similar in pandas and sql!
        col_sel = [
                'recipesID', 'title', 'ingredients', 'rating', 'calories',
                'sodium', 'fat', 'protein', 'ghg', 'prop_ing', 'ghg_log10',
                'url', 'servings', 'index', 'image_url', 'perc_rating',
                'perc_sustainability', 'review_count'
                    ]
        recipes_sql = self.query_similar_recipes(CS_ids)

        # Obtain a dataframe for further processing
        results = pd.DataFrame(recipes_sql, columns=col_sel)

        # Add similarity scores to correct recipes (using recipesID again)
        temp = pd.DataFrame({'CS_ids': CS_ids, 'similarity': CS})
        results = results.merge(temp, left_on='recipesID',
                                right_on='CS_ids', how='left')

        # Assign data types (sql output might be decimal, should
        # be float!)
        numerics = ['recipesID', 'rating', 'calories', 'sodium',
                    'fat', 'protein', 'ghg', 'prop_ing', 'ghg_log10',
                    'index', 'perc_rating', 'perc_sustainability',
                    'similarity', 'review_count']
        strings = ['title', 'ingredients', 'url', 'servings', 'image_url']
        for num in numerics:
            results[num] = pd.to_numeric(results[num])
        for s in strings:
            results[s] = results[s].astype('str')

        # Order results by similarity
        results = results.sort_values(by='similarity', ascending=False)

        return results

    def search_recipes(self, search_term):
        """
        DESCRIPTION:
            Does a fuzzy search for recipes based on user's search term. If an
            exact match exists, does a content based search and returns the
            resulting DataFrame. If no exact match exists, return a DataFrame
            with the fuzzily matched search results.
        INPUT:
            cur: psycopg2 cursor object
            search_term (str): Search term input by user into search bar
        OUTPUT:
            df (pd.DataFrame): DataFrame with recipes as rows
        """
        outp = self.fuzzy_search(search_term)

        if outp:
            if outp[0][2] == search_term:
                return self.content_based_search(search_term)

        col_names = ["recipesID", "title", "url", "perc_rating",
                     "perc_sustainability", "review_count", "image_url",
                     "ghg", "prop_ingredients", "edit_dist"]

        results = pd.DataFrame(outp, columns=col_names)

        # Assign data types (sql output might be decimal, should
        # be float!)
        numerics = ['recipesID', 'perc_rating', 'ghg', 'prop_ingredients',
                    'perc_rating', 'perc_sustainability', 'review_count']
        strings = ['title', 'url', 'image_url']
        for num in numerics:
            results[num] = pd.to_numeric(results[num])
        for s in strings:
            results[s] = results[s].astype('str')

        # Order results by edit_dist
        results = results.sort_values(by='edit_dist', ascending=True)
        return results


# eof



# Create postgres DB connection object
pg = postgresConnection()

# Exact search
search_term = 'pineapple-shrimp-noodle-bowls'
results = pg.content_based_search(search_term)
results.head()

# Check if I can query something fuzzily:
results = pg.search_recipes('chicken')
results.head(100)
print(pg.search_recipes('chicken'))


## eof






















