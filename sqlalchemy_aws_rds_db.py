# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:25:05 2020

Trying to set up a postgres SQL database for sustainable-recipe-recomender
using AWS RDS.

@author: sbuer
"""

# Mostly following:
# https://towardsdatascience.com/how-to-set-up-a-postgresql-database-on-amazon-rds-64e8d144179e


# for loading environment variables
import os

# needed to load environment variables from .env file
from dotenv import load_dotenv 
load_dotenv()

# flask and flask_sqlalchemy
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, bindparam, String, Integer, Numeric

# data handling
import pandas as pd
import datetime

# code testing
import pytest


# instantiate Flask application with flask_sqlalchemy database
load_dotenv('.env')
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_NATIVE_UNICODE'] = True
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True
}
db = SQLAlchemy(app)
db.Model.metadata.reflect(db.engine)


class User(db.Model):
    __table__ = db.Model.metadata.tables['users']
    likes = db.relationship('Like', backref='user', lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)


class Recipe(db.Model):
    __table__ = db.Model.metadata.tables['recipes']

    def __repr__(self):
        return '<Recipe {}>'.format(self.title)


class Like(db.Model):
    __table__ = db.Model.metadata.tables['likes']

    def __repr__(self):
        return '<Like {}>'.format(self.likeID)


class Consent(db.Model):
    __table__ = db.Model.metadata.tables['consent']

    def __repr__(self):
        return '<Consent {}>'.format(self.consentID)


class ContentSimilarity(db.Model):
    __table__ = db.Model.metadata.tables['content_similarity200']

    def __repr__(self):
        return '<ContentSimilarity {}>'.format(self.recipeID)


class ContentSimilarityID(db.Model):
    __table__ = db.Model.metadata.tables['content_similarity200_ids']

    def __repr__(self):
        return '<ContentSimilarityID {}>'.format(self.recipeID)


def fuzzy_search(session, search_term, search_column="url", N=160):
    """
    DESCRIPTION:
        Searches in recipes table column url for strings that include the
        search_term. If none do, returns the top N results ordered
        by edit distance in ascending order.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str): String to look for in search_column
        search_column (str): Column to search (default="url")
        N (int): Max number of results to return
    OUTPUT:
        results (list of RowProxy objects): query results
    """
    # Most similar urls by edit distance that actually contain the
    # search_term
    query = text(
        """
        SELECT "recipesID", "title", "url", "perc_rating",
            "perc_sustainability", "review_count", "image_url",
            "emissions", "prop_ingredients",
            LEVENSHTEIN(:search_column, :search_term) AS "rank"
        FROM public.recipes
        WHERE :search_column LIKE :search_term_like
        ORDER BY "rank" ASC
        LIMIT :N
        """,
        bindparams=[
            bindparam('search_column', value=search_column, type_=String),
            bindparam('search_term', value=search_term, type_=String),
            bindparam('search_term_like', value='%'+search_term+'%',
                      type_=String),
            bindparam('N', value=N, type_=Integer)
        ]
    )
    results = session.execute(query).fetchall()

    # If no results contain the search_term
    if not results:
        query = text(
            """
            SELECT "recipesID", "title", "url", "perc_rating",
                "perc_sustainability", "review_count", "image_url",
                "emissions", "prop_ingredients",
                LEVENSHTEIN(:search_column, :search_term) AS "rank"
            FROM public.recipes
            ORDER BY "rank" ASC
            LIMIT :N
            """,
            bindparams=[
                bindparam('search_column', value=search_column, type_=String),
                bindparam('search_term', value=search_term, type_=String),
                bindparam('N', value=N, type_=Integer)
            ]
        )
        results = session.execute(query).fetchall()
    return results


def phrase_search(session, search_term, N=160):
    """
    DESCRIPTION:
        Searches in table recipes in combined_tsv column using tsquery
        - a tsvector column in DB table recipes combining title and
        categories.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str): Search term
        N (int): Max number of results to return
    OUTPUT:
        results (list of RowProxy objects): DB query result
    """
    query = text(
        """
        SELECT "recipesID", "title", "url", "perc_rating",
            "perc_sustainability", "review_count", "image_url",
            "emissions", "prop_ingredients",
            ts_rank_cd(combined_tsv, query) AS "rank"
        FROM public.recipes,
            websearch_to_tsquery('simple', :search_term) query
        WHERE query @@ combined_tsv
        ORDER BY "rank" DESC
        LIMIT :N
        """,
        bindparams=[
            bindparam('search_term', value=search_term, type_=String),
            bindparam('N', value=N, type_=Integer)
        ]
    )
    results = session.execute(query).fetchall()
    return results


def free_search(session, search_term, N=160):
    """
    DESCRIPTION:
        Parent function for searching recipes freely. At the moment
        it only calls phrase_search. But having this function makes
        it easier to extend in the future.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str)
        N (int): Max number of results to return
    OUTPUT:
        results (list of RowProxy objects): DB query result
    NOTES:
        See https://www.postgresql.org/docs/12/textsearch-controls.html
        for details on postgres' search functionalities.
    """
    results = phrase_search(session, search_term, N=N)
    if not results:
        results = fuzzy_search(session, search_term, N=N-len(results))
    return results


def query_content_similarity_ids(session, search_term):
    """
    DESCRIPTION:
        Searches in connected postgres DB for a search_term in
        'url' column and returns recipeIDs of similar recipes based
        on content similarity.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str): Search term
    OUTPUT:
        CS_ids (tuple): Content based similarity ID vector ordered by
            similarity in descending order
    """
    query = text(
        """
        SELECT * FROM public.content_similarity200_ids
        WHERE "recipeID" = (
            SELECT "recipesID" FROM public.recipes
            WHERE "url" = :search_term)
        """,
        bindparams=[
            bindparam('search_term', value=search_term, type_=String)
        ]
    )
    CS_ids = session.execute(query).fetchall()[0][1::]
    CS_ids = tuple([abs(int(CSid)) for CSid in CS_ids])
    return CS_ids


def query_content_similarity(session, search_term):
    """
    DESCRIPTION:
        Searches in connected postgres DB for a search_term in
        'url' and returns content based similarity.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str): Search term
    OUTPUT:
        CS (tuple): Content based similarity vector ordered by
            similarity in descending order
    """
    query = text(
        """
        SELECT * FROM public.content_similarity200
        WHERE "recipeID" = (
            SELECT "recipesID" FROM public.recipes
            WHERE url = :search_term)
        """,
        bindparams=[
            bindparam('search_term', value=search_term, type_=String)
        ]
    )
    CS = session.execute(query).fetchall()[0][1::]
    CS = tuple([abs(float(s)) for s in CS])
    return CS


def query_similar_recipes(session, CS_ids):
    """
    DESCRIPTION:
        fetch recipe information of similar recipes based on the recipe IDs
        given by CS_ids
    INPUT:
        session: (Flask-)SQLAlchemy session object
        CS_ids (tuple): Tuple of recipe IDs
    OUTPUT:
        recipes_sql (list of RowProxy objects): DB query result
    """
    query = text(
        """
        SELECT "recipesID", "title", "ingredients",
            "rating", "calories", "sodium", "fat",
            "protein", "emissions", "prop_ingredients",
            "emissions_log10", "url", "servings", "recipe_rawid",
            "image_url", "perc_rating", "perc_sustainability",
            "review_count"
        FROM public.recipes
        WHERE "recipesID" IN :CS_ids
        """,
        bindparams=[
            bindparam('CS_ids', value=CS_ids, type_=Numeric)
        ]
    )
    recipes_sql = session.execute(query).fetchall()
    return recipes_sql


def exact_recipe_match(session, search_term):
    '''
    DESCRIPTION:
        Return True if search_term is in recipes table of
        cur database, False otherwise.
    '''
    query = text(
        """
        SELECT * FROM public.recipes
        WHERE "url" = :search_term
        """,
        bindparams=[
            bindparam('search_term', value=search_term, type_=String)
        ]
    )
    if session.execute(query).fetchall():
        return True
    else:
        return False


def content_based_search(session, search_term):
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
    CS_ids = query_content_similarity_ids(session, search_term)

    # Also select the actual similarity scores
    CS = query_content_similarity(session, search_term)

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
    recipes_sql = query_similar_recipes(session, CS_ids)

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


def search_recipes(session, search_term, N=160):
    """
    DESCRIPTION:
        Does a free search for recipes based on user's search term. If an
        exact match exists, does a content based search and returns the
        resulting DataFrame.
    INPUT:
        session: (Flask-)SQLAlchemy session object
        search_term (str): Search term input by user into search bar
        N (int): Max number of results to return
    OUTPUT:
        df (pd.DataFrame): DataFrame with recipes as rows
    """
    outp = free_search(session, search_term, N)

    if outp[0][2] == search_term:
        return content_based_search(session, search_term)

    col_names = ["recipesID", "title", "url", "perc_rating",
                 "perc_sustainability", "review_count", "image_url",
                 "ghg", "prop_ingredients", "rank"]

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

    # Order results by rank / edit_dist
    results = results.sort_values(by='rank', ascending=False)
    return results


def query_cookbook(session, userID):
    """
    DESCRIPTION:
        Creates a pandas dataframe containing all recipes the given
        user has liked / added to the cookbook.
    INPUT:
        userID (Integer)
    OUTPUT:
        cookbook (pd.DataFrame)
    """
    query = text(
        """
        SELECT u."userID", u.username,
            l.created, l.rating,
            r.title, r.url, r.perc_rating, r.perc_sustainability,
            r.review_count, r.image_url, r.emissions, r.prop_ingredients,
            r.categories
            FROM users u
            JOIN likes l ON (u.username = l.username)
            JOIN recipes r ON (l."recipesID" = r."recipesID")
        WHERE u."userID" = :userID
        ORDER BY l.rating
        """,
        bindparams=[
            bindparam('userID', value=userID, type_=Integer)
        ]
    )
    recipes = session.execute(query).fetchall()

    # Convert to DataFrame
    col_sel = ["userID", "username", "created", "user_rating",
               "recipe_title", "url", "perc_rating", "perc_sustainability",
               "review_count", "image_url", "emissions", "prop_ingredients",
               "categories"]
    results = pd.DataFrame(recipes, columns=col_sel)

    # Assign data types
    numerics = ['userID', 'user_rating', 'perc_rating', 'perc_sustainability',
                'review_count', 'emissions', 'prop_ingredients']
    strings = ['username', 'recipe_title', 'url', 'image_url', 'categories']
    datetimes = ['created']
    for num in numerics:
        results[num] = pd.to_numeric(results[num])
    for s in strings:
        results[s] = results[s].astype('str')
    for dt in datetimes:
        results[dt] = pd.to_datetime(results[dt])
    return results


def is_in_cookbook(session, userID, url):
    """
    DESCRIPTION:
        Check if a recipe (given by url) is already in a user's
        cookbook (given by userID)
    INPUT:
        userID (Integer): userID from users table
        url (String): Url string from recipes table
    OUTPUT:
        Boolean
    """
    # Get username and recipesID
    recipe = Recipe.query.filter_by(url=url).first()

    # Query like entries
    like = Like.query.filter_by(userID=userID,
                                recipesID=recipe.recipesID).first()
    if like:
        return True
    return False


def add_to_cookbook(session, userID, url):
    """
    DESCRIPTION:
        Creates a new entry in the likes table for a given user
        and recipe.
    INPUT:
        userID (Integer): userID from users table
        url (String): Url string from recipes table
    OUTPUT:
        String: Feedback message
    """
    # Get username and recipesID
    user = User.query.filter_by(userID=userID).first()
    recipe = Recipe.query.filter_by(url=url).first()

    # Create new like entry
    like = Like(username=user.username,
                rating=None,
                userID=userID,
                recipesID=recipe.recipesID,
                created=datetime.datetime.utcnow())
    session.add(like)
    session.commit()


def remove_from_cookbook(session, userID, url):
    """
    DESCRIPTION:
        Removes an existing entry in the likes table for a given
        user and recipe.
    INPUT:
        userID (Integer): userID from users table
        url (String): Url string from recipes table
    OUTPUT:
        String: Feedback message
    """
    # Get like entry based on userID and recipe url
    recipe = Recipe.query.filter_by(url=url).first()
    like = Like.query.filter_by(userID=userID,
                                recipesID=recipe.recipesID).first()

    # Create new like entry
    session.delete(like)
    session.commit()


def query_user_ratings(session, userID, urls):
    """
    DESCRIPTION:
        Query all rows in likes table with the given userID
        for all elements in urls
    INPUT:
        userID (Integer): userID from users table
        urls (List of strings): Url strings from recipes table
    OUTPUT:
        pandas.DataFrame with columns [likeID, userID, recipesID,
                username, bookmarked, rating, created], can be empty
    """
    recipesIDs = session.query(Recipe.recipesID).\
        filter(Recipe.url.in_(urls)).all()
    likes_query = session.query(Like).\
        filter(Like.userID == userID,
               Like.recipesID.in_(recipesIDs))
    df = pd.read_sql(likes_query.statement, session.bind)
    df.rename(columns={'rating': 'user_rating'}, inplace=True)
    return df


userID = 3
urls = ['bla-blub', 'blabla-blubblub', 'pineapple-shrimp-noodle-bowls',
        'cold-sesame-noodles-12715']
query_user_ratings(db.session, userID, urls)

urls = ['bla-blub', 'blabla-blubblub']
query_user_ratings(db.session, userID, urls)


def rate_recipe(session, userID, url, rating):
    """
    DESCRIPTION:
        Add user rating to bookmarked recipe in DB.
    INPUT:
        userID (Integer): userID from users table
        url (String): Recipe url tag
    OUTPUT:
        None
    """
    # Get recipeID
    recipeID = session.query(Recipe.recipesID).\
        filter(Recipe.url == url).first()

    # Find relevant likes row
    like = Like.query.filter_by(userID=userID, recipesID=recipeID).first()

    # Add user rating and commit to DB
    like.rating = rating
    session.add(like)
    session.commit()


# TEST rate_recipe
url = 'pineapple-shrimp-noodle-bowls'

# Dislike a recipe
rate_recipe(db.session, userID, url, rating=1)
df = query_user_ratings(db.session, userID, [url])
assert df['rating'][0] == 1

# Like a recipe
rate_recipe(db.session, userID, url, rating=5)
df = query_user_ratings(db.session, userID, [url])
assert df['rating'][0] == 5


# Query recipes given by urls, and for this particular user return which
# ones are bookmarked
userID = 3
urls = ['bla-blub', 'blabla-blubblub', 'pineapple-shrimp-noodle-bowls',
        'cold-sesame-noodles-12715']
sql_query = db.session.query(
    Recipe, Like
).join(
    Like, Like.recipesID == Recipe.recipesID, isouter=True
).filter(
    Like.userID == userID,
    Recipe.url.in_(urls)
)
df = pd.read_sql(sql_query.statement, db.session.bind)
df


def query_bookmarks(session, userID, urls):
    """
    DESCRIPTION:
        For all recipes (given in list urls) check if it has
        been bookmarked by the user (return boolean list).
    INPUT:
        userID (Integer): userID from users table
        urls (List of strings): Url strings from recipes table
    OUTPUT:
        Pandas DataFrame with columns 'recipesID' and 'bookmarked'
    """
    sql_query = session.query(
        Recipe, Like
    ).join(
        Like, Like.recipesID == Recipe.recipesID, isouter=True
    ).filter(
        Like.userID == userID,
        Recipe.url.in_(urls)
    )
    df = pd.read_sql(sql_query.statement, session.bind)

    # I got 2 recipeID columns, keep only one!
    df = df.loc[:, ~df.columns.duplicated()]
    return df[['recipesID', 'bookmarked']]


def delete_account(session, userID):
    """
    DESCRIPTION:
        Removes an existing user entry from users table,
        and corresponding rows in consent and likes tables.
    INPUT:
        userID (Integer): userID from users table
    OUTPUT:
        String: Feedback message
    """
    user = User.query.filter_by(userID=userID).first()
    if user:

        # delete likes of user (in likes table)
        Like.query.filter_by(userID=userID).delete()

        # delete consent of user (in consent table)
        Consent.query.filter_by(userID=userID).delete()

        # delete user (in users table)
        session.delete(user)
        session.commit()
        return 'Removed user account successfully'
    return 'User not found. Nothing was removed.'


# Get results and corresponding booksmarks (see compare_recipes route)
search_term = 'pineapple-shrimp-noodle-bowls'
results = content_based_search(db.session, search_term)

# Disentangle reference recipe and similar recipes
ref_recipe = results.iloc[0]
results = results.iloc[1::, :]

# Select only the top Np recipes for one page
page = 1
Np = 20
results = results[(0+page*Np):((page+1)*Np)]

#
urls = results['url']
df_bookmarks = query_bookmarks(db.session, userID, urls)
results = results.merge(df_bookmarks, how='left', on='recipesID')

# Replace NaNs with False in bookmarked column
results['bookmarked'].fillna(False, inplace=True)

results

# -------------

# Check combination of search_results and user's liked recipes
# (e.g. in compare_recipes)

# Step 1: Recipe search results for search_term
search_term = 'pineapple-shrimp-noodle-bowls'
results = content_based_search(db.session, search_term)

# Step 2: For this user and those recipes, which ones are liked?
userID = 3
urls = results['url']
user_results = query_user_ratings(db.session, userID, urls)

# Step 3: Combine data
test = results.merge(user_results[['user_rating', 'recipesID']], how='left', on='recipesID')

# -------------

# Query all recipe emissions (for histgoram_emissions figure)
recipes = db.session.query(Recipe.recipesID, Recipe.emissions_log10, Recipe.url, Recipe.title)
df = pd.read_sql(recipes.statement, db.session.bind)
df.rename(columns={'emissions_log10': 'Emissions'}, inplace=True)


# -------------

# Query categories for a selection of recipes and order them by frequency
userID = 3
df = query_cookbook(db.session, userID)


def get_favorite_categories(df):
    category_list = []
    for item in df['categories']:
        category_list.extend(item.split(';'))
    count_table = [(l, category_list.count(l)) for l in set(category_list)]	
    count_table.sort(reverse=True, key=lambda x: x[1])	
    labels = [item[0] for item in count_table]
    counts = [item[1] for item in count_table]
    return labels, counts

# ------------

# show me all columns in a model
def show_model_columns(model):
    '''
    Show all columns in sql alchemy model with name "model"
    '''
    from sqlalchemy.inspection import inspect
    return [column.name for column in inspect(User).c]

    show_model_columns(User)


# eof
