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

# for connecting to postgres database
import psycopg2 as ps
from psycopg2 import sql


# create connection and cursor   
def connect_to_DB(): 
	conn = ps.connect(host=os.environ.get('AWS_POSTGRES_ADDRESS'),
	                  database=os.environ.get('AWS_POSTGRES_DBNAME'),
	                  user=os.environ.get('AWS_POSTGRES_USERNAME'),
	                  password=os.environ.get('AWS_POSTGRES_PASSWORD'),
	                  port=os.environ.get('AWS_POSTGRES_PORT'))
	cur = conn.cursor()		
	return conn, cur


# check DB content
def check_db_content(cur):
	query = """SELECT * FROM pg_catalog.pg_tables
	            WHERE schemaname != 'pg_catalog'
	            AND schemaname != 'information_schema';"""
	cur.execute(query)
	return cur.fetchall()

# check table columns
def check_table_columns(cur, table_name):
	query = """ SELECT * FROM information_schema.columns
				WHERE table_name = 'recipes';"""
	cur.execute(query)
	outp = cur.fetchall()
	return [o[3] for o in outp]


# Show DB info
conn, cur = connect_to_DB()
check_db_content(cur)


# Show columns of recipes table
check_table_columns(cur, 'recipes')


# Create recipes Table
# TODO: Why assign character ranges to VARCHAR?!
cur.execute(
	'''
	-- Table: public.recipes

	-- DROP TABLE public.recipes;
	
	CREATE TABLE public.recipes
	(
	    "recipesID" bigint NOT NULL,
	    title character varying(250) COLLATE pg_catalog."default",
	    ingredients character varying(4000) COLLATE pg_catalog."default",
	    categories character varying(5000) COLLATE pg_catalog."default",
	    date date,
	    rating numeric,
	    calories numeric,
	    sodium numeric,
	    fat numeric,
	    protein numeric,
	    emissions numeric,
	    prop_ingredients numeric,
	    emissions_log10 numeric,
	    url character varying(1000) COLLATE pg_catalog."default",
	    servings character varying(1000) COLLATE pg_catalog."default",
	    recipe_rawid bigint NOT NULL,
	    image_url character varying(1000) COLLATE pg_catalog."default",
	    perc_rating numeric,
	    perc_sustainability numeric,
		review_count numeric,
		
	    CONSTRAINT recipes_pkey PRIMARY KEY ("recipesID")
	)
	WITH (
	    OIDS = FALSE
	)
	TABLESPACE pg_default;
	
	ALTER TABLE public.recipes
	    OWNER to postgres;
	COMMENT ON TABLE public.recipes
	    IS 'each row is a recipe';
	''')
	
# Commit table creation
conn.commit()


# Create content_similarity200 table
cur.execute(
	'''
	CREATE TABLE public.content_similarity200
	(
	    "recipeID" bigint NOT NULL,
	    "0" numeric,
	    "1" numeric,
	    "2" numeric,
	    "3" numeric,
	    "4" numeric,
	    "5" numeric,
	    "6" numeric,
	    "7" numeric,
	    "8" numeric,
	    "9" numeric,
	    "10" numeric,
	    "11" numeric,
	    "12" numeric,
	    "13" numeric,
	    "14" numeric,
	    "15" numeric,
	    "16" numeric,
	    "17" numeric,
	    "18" numeric,
	    "19" numeric,
	    "20" numeric,
	    "21" numeric,
	    "22" numeric,
	    "23" numeric,
	    "24" numeric,
	    "25" numeric,
	    "26" numeric,
	    "27" numeric,
	    "28" numeric,
	    "29" numeric,
	    "30" numeric,
	    "31" numeric,
	    "32" numeric,
	    "33" numeric,
	    "34" numeric,
	    "35" numeric,
	    "36" numeric,
	    "37" numeric,
	    "38" numeric,
	    "39" numeric,
	    "40" numeric,
	    "41" numeric,
	    "42" numeric,
	    "43" numeric,
	    "44" numeric,
	    "45" numeric,
	    "46" numeric,
	    "47" numeric,
	    "48" numeric,
	    "49" numeric,
	    "50" numeric,
	    "51" numeric,
	    "52" numeric,
	    "53" numeric,
	    "54" numeric,
	    "55" numeric,
	    "56" numeric,
	    "57" numeric,
	    "58" numeric,
	    "59" numeric,
	    "60" numeric,
	    "61" numeric,
	    "62" numeric,
	    "63" numeric,
	    "64" numeric,
	    "65" numeric,
	    "66" numeric,
	    "67" numeric,
	    "68" numeric,
	    "69" numeric,
	    "70" numeric,
		"71" numeric,
	    "72" numeric,
	    "73" numeric,
	    "74" numeric,
	    "75" numeric,
	    "76" numeric,
	    "77" numeric,
	    "78" numeric,
	    "79" numeric,
	    "80" numeric,
		"81" numeric,
	    "82" numeric,
	    "83" numeric,
	    "84" numeric,
	    "85" numeric,
	    "86" numeric,
	    "87" numeric,
	    "88" numeric,
	    "89" numeric,
	    "90" numeric,
		"91" numeric,
	    "92" numeric,
	    "93" numeric,
	    "94" numeric,
	    "95" numeric,
	    "96" numeric,
	    "97" numeric,
	    "98" numeric,
	    "99" numeric,
	    "100" numeric,
	    "101" numeric,
	    "102" numeric,
	    "103" numeric,
	    "104" numeric,
	    "105" numeric,
	    "106" numeric,
	    "107" numeric,
	    "108" numeric,
	    "109" numeric,
	    "110" numeric,
	    "111" numeric,
	    "112" numeric,
	    "113" numeric,
	    "114" numeric,
	    "115" numeric,
	    "116" numeric,
	    "117" numeric,
	    "118" numeric,
	    "119" numeric,
	    "120" numeric,
	    "121" numeric,
	    "122" numeric,
	    "123" numeric,
	    "124" numeric,
	    "125" numeric,
	    "126" numeric,
	    "127" numeric,
	    "128" numeric,
	    "129" numeric,
	    "130" numeric,
	    "131" numeric,
	    "132" numeric,
	    "133" numeric,
	    "134" numeric,
	    "135" numeric,
	    "136" numeric,
	    "137" numeric,
	    "138" numeric,
	    "139" numeric,
	    "140" numeric,
	    "141" numeric,
	    "142" numeric,
	    "143" numeric,
	    "144" numeric,
	    "145" numeric,
	    "146" numeric,
	    "147" numeric,
	    "148" numeric,
	    "149" numeric,
	    "150" numeric,
	    "151" numeric,
	    "152" numeric,
	    "153" numeric,
	    "154" numeric,
	    "155" numeric,
	    "156" numeric,
	    "157" numeric,
	    "158" numeric,
	    "159" numeric,
	    "160" numeric,
	    "161" numeric,
	    "162" numeric,
	    "163" numeric,
	    "164" numeric,
	    "165" numeric,
	    "166" numeric,
	    "167" numeric,
	    "168" numeric,
	    "169" numeric,
	    "170" numeric,
		"171" numeric,
	    "172" numeric,
	    "173" numeric,
	    "174" numeric,
	    "175" numeric,
	    "176" numeric,
	    "177" numeric,
	    "178" numeric,
	    "179" numeric,
	    "180" numeric,
		"181" numeric,
	    "182" numeric,
	    "183" numeric,
	    "184" numeric,
	    "185" numeric,
	    "186" numeric,
	    "187" numeric,
	    "188" numeric,
	    "189" numeric,
	    "190" numeric,
		"191" numeric,
	    "192" numeric,
	    "193" numeric,
	    "194" numeric,
	    "195" numeric,
	    "196" numeric,
	    "197" numeric,
	    "198" numeric,
	    "199" numeric,
	    PRIMARY KEY ("recipeID")
	)
	WITH (
	    OIDS = FALSE
	);
	
	ALTER TABLE public.content_similarity200
	    OWNER to postgres;
	''')
	
# Commit table creation
conn.commit()


# I insert the data using pgadmin4 for now, but I could insert data from here
# using:
'''
data = [[5, 5.5, 'five', True], [5, 5.5, 'five', True], [5, 5.5, 'five', True]]
insert_query = """INSERT INTO table_1
                   (column_1, column_2, column_3, column_4)
                   VALUES (%s, %s, %s, %s);"""
# execute multiple inserts
cur.executemany(insert_query, data)
            
# commit data insert
conn.commit()
'''

# Delete or drop a table
'''
cur.execute("""DROP TABLE table_1""")
conn.commit()
'''


# fetch some data from table public.content_similarity200
cur.execute('''
			 SELECT * FROM public.content_similarity200
			 LIMIT 10
		    ''')
cur.fetchall()


# Create table for content_similarity200_IDs
cur.execute(
	'''
	CREATE TABLE public.content_similarity200_IDs
	(
	    "recipeID" bigint NOT NULL,
	    "0" bigint,
	    "1" bigint,
	    "2" bigint,
	    "3" bigint,
	    "4" bigint,
	    "5" bigint,
	    "6" bigint,
	    "7" bigint,
	    "8" bigint,
	    "9" bigint,
	    "10" bigint,
	    "11" bigint,
	    "12" bigint,
	    "13" bigint,
	    "14" bigint,
	    "15" bigint,
	    "16" bigint,
	    "17" bigint,
	    "18" bigint,
	    "19" bigint,
	    "20" bigint,
	    "21" bigint,
	    "22" bigint,
	    "23" bigint,
	    "24" bigint,
	    "25" bigint,
	    "26" bigint,
	    "27" bigint,
	    "28" bigint,
	    "29" bigint,
	    "30" bigint,
	    "31" bigint,
	    "32" bigint,
	    "33" bigint,
	    "34" bigint,
	    "35" bigint,
	    "36" bigint,
	    "37" bigint,
	    "38" bigint,
	    "39" bigint,
	    "40" bigint,
	    "41" bigint,
	    "42" bigint,
	    "43" bigint,
	    "44" bigint,
	    "45" bigint,
	    "46" bigint,
	    "47" bigint,
	    "48" bigint,
	    "49" bigint,
	    "50" bigint,
	    "51" bigint,
	    "52" bigint,
	    "53" bigint,
	    "54" bigint,
	    "55" bigint,
	    "56" bigint,
	    "57" bigint,
	    "58" bigint,
	    "59" bigint,
	    "60" bigint,
	    "61" bigint,
	    "62" bigint,
	    "63" bigint,
	    "64" bigint,
	    "65" bigint,
	    "66" bigint,
	    "67" bigint,
	    "68" bigint,
	    "69" bigint,
	    "70" bigint,
		"71" bigint,
	    "72" bigint,
	    "73" bigint,
	    "74" bigint,
	    "75" bigint,
	    "76" bigint,
	    "77" bigint,
	    "78" bigint,
	    "79" bigint,
	    "80" bigint,
		"81" bigint,
	    "82" bigint,
	    "83" bigint,
	    "84" bigint,
	    "85" bigint,
	    "86" bigint,
	    "87" bigint,
	    "88" bigint,
	    "89" bigint,
	    "90" bigint,
		"91" bigint,
	    "92" bigint,
	    "93" bigint,
	    "94" bigint,
	    "95" bigint,
	    "96" bigint,
	    "97" bigint,
	    "98" bigint,
	    "99" bigint,
	    "100" bigint,
	    "101" bigint,
	    "102" bigint,
	    "103" bigint,
	    "104" bigint,
	    "105" bigint,
	    "106" bigint,
	    "107" bigint,
	    "108" bigint,
	    "109" bigint,
	    "110" bigint,
	    "111" bigint,
	    "112" bigint,
	    "113" bigint,
	    "114" bigint,
	    "115" bigint,
	    "116" bigint,
	    "117" bigint,
	    "118" bigint,
	    "119" bigint,
	    "120" bigint,
	    "121" bigint,
	    "122" bigint,
	    "123" bigint,
	    "124" bigint,
	    "125" bigint,
	    "126" bigint,
	    "127" bigint,
	    "128" bigint,
	    "129" bigint,
	    "130" bigint,
	    "131" bigint,
	    "132" bigint,
	    "133" bigint,
	    "134" bigint,
	    "135" bigint,
	    "136" bigint,
	    "137" bigint,
	    "138" bigint,
	    "139" bigint,
	    "140" bigint,
	    "141" bigint,
	    "142" bigint,
	    "143" bigint,
	    "144" bigint,
	    "145" bigint,
	    "146" bigint,
	    "147" bigint,
	    "148" bigint,
	    "149" bigint,
	    "150" bigint,
	    "151" bigint,
	    "152" bigint,
	    "153" bigint,
	    "154" bigint,
	    "155" bigint,
	    "156" bigint,
	    "157" bigint,
	    "158" bigint,
	    "159" bigint,
	    "160" bigint,
	    "161" bigint,
	    "162" bigint,
	    "163" bigint,
	    "164" bigint,
	    "165" bigint,
	    "166" bigint,
	    "167" bigint,
	    "168" bigint,
	    "169" bigint,
	    "170" bigint,
		"171" bigint,
	    "172" bigint,
	    "173" bigint,
	    "174" bigint,
	    "175" bigint,
	    "176" bigint,
	    "177" bigint,
	    "178" bigint,
	    "179" bigint,
	    "180" bigint,
		"181" bigint,
	    "182" bigint,
	    "183" bigint,
	    "184" bigint,
	    "185" bigint,
	    "186" bigint,
	    "187" bigint,
	    "188" bigint,
	    "189" bigint,
	    "190" bigint,
		"191" bigint,
	    "192" bigint,
	    "193" bigint,
	    "194" bigint,
	    "195" bigint,
	    "196" bigint,
	    "197" bigint,
	    "198" bigint,
	    "199" bigint,
	    PRIMARY KEY ("recipeID")
	)
	WITH (
	    OIDS = FALSE
	);
	
	ALTER TABLE public.content_similarity200
	    OWNER to postgres;
	''')
	
# Commit table creation
conn.commit()


# Check if data upload (from pgadmin4) worked
cur.execute('''
			 SELECT * FROM public.content_similarity200_IDs
			 LIMIT 10
		    ''')
cur.fetchall()

# ===========================================================================
# Add column to recipes table containing tsvector elements
# of the recipe "title" column. Will make searching the DB
# with user input much easier.
# See https://www.compose.com/articles/mastering-postgresql-tools-full-text-search-and-phrase-search/

# Create temporary table to test following code
def make_temp_table(cur, conn):
	cur.execute(
		'''
		CREATE TABLE public.test_table
		(
			"recipesID" bigint NOT NULL,
			title VARCHAR,
			title_tsv TSVECTOR,
			PRIMARY KEY ("recipesID")
		)
		
		WITH (
		    OIDS = FALSE
		)
		TABLESPACE pg_default;
		
		ALTER TABLE public.test_table
		OWNER to postgres;
		''')
	conn.commit()

# Drop temporary table
def drop_temp_table(cur, conn):
	cur.execute(''' DROP TABLE IF EXISTS public.test_table; ''')
	conn.commit()

# Create temp table, 
# check if it exists, 
# then drop it again.
# run step by step to see output
make_temp_table(cur, conn)
check_db_content(cur)
drop_temp_table(cur, conn)
check_db_content(cur)

# Create temp table
make_temp_table(cur, conn)
check_db_content(cur)

# Check new test_table
cur.execute('''
			SELECT column_name, data_type 
			FROM information_schema.columns 
			WHERE table_name = 'test_table';
		    ''')
cur.fetchall()

cur.execute('''
			SELECT * FROM test_table
			LIMIT 1
		    ''')
cur.fetchall()


# update test_table (fill with data from recipes table)
cur.execute('''
			INSERT INTO test_table ("recipesID", "title")
			SELECT "recipesID", "title"
			FROM recipes;
			''')
conn.commit()

# show inserted data and overall size
cur.execute('''
			SELECT * FROM test_table
			LIMIT 10
		    ''')
cur.fetchall()

cur.execute(''' SELECT count(*) FROM test_table ''')
cur.fetchall()

cur.execute(''' SELECT count(*) FROM recipes ''')
cur.fetchall()

# Add tsvector column of "title" column
cur.execute('''
			UPDATE test_table
			SET title_tsv = to_tsvector(title);
			''')
conn.commit()

# show data
cur.execute('''
			SELECT * FROM test_table
			LIMIT 10
		    ''')
cur.fetchall()

# drop test table
drop_temp_table(cur, conn)
check_db_content(cur)


# Add tsvector column of "title" column in recipes table
# Show current column names
cur.execute('''
			SELECT column_name, data_type 
			FROM information_schema.columns 
			WHERE table_name = 'recipes';
		    ''')
cur.fetchall()

# Add new column: title_tsv
cur.execute('''
			ALTER TABLE recipes
			ADD COLUMN title_tsv TSVECTOR;
		    ''')
conn.commit()

# Check if new column is there
cur.execute('''
			SELECT column_name, data_type 
			FROM information_schema.columns 
			WHERE table_name = 'recipes';
		    ''')
cur.fetchall()
			
# Populate title_tsv
cur.execute('''
			UPDATE recipes
			SET title_tsv = to_tsvector(title);
			''')
conn.commit()

# show data
cur.execute('''
			SELECT * FROM recipes
			LIMIT 10
		    ''')
cur.fetchall()


# check output of websearch_to_tsquery()
cur.execute(sql.SQL(
            """
            SELECT websearch_to_tsquery('simple', '"vegan cookies"');
            """))
cur.fetchall()

# Test free search with tsquery
search_term = 'vegan cookies'
N = 10
cur.execute(sql.SQL(
            """
            SELECT "recipesID", "title", "title_tsv",
                ts_rank_cd(title_tsv, query) AS rank
            FROM public.recipes, websearch_to_tsquery('simple', %s) query
            WHERE query @@ title_tsv
            ORDER BY rank ASC
            LIMIT %s
            """).format(), [search_term, N])
cur.fetchall()


# ===========================================================================
# Add tsvector columns to recipes table for 
# 1. categories
# 2. ingredients

# Show columns of recipes table
check_table_columns(cur, 'recipes')

# Create temporary DB for testing
def make_temp_table2(cur, conn):
	cur.execute(
		'''
		CREATE TABLE public.test_table
		(
			"recipesID" bigint NOT NULL,
			categories VARCHAR,
			categories_tsv TSVECTOR,
			ingredients VARCHAR,
			ingredients_tsv TSVECTOR,
			PRIMARY KEY ("recipesID")
		)
		
		WITH (
		    OIDS = FALSE
		)
		TABLESPACE pg_default;
		
		ALTER TABLE public.test_table
		OWNER to postgres;
		''')
	conn.commit()
	
	
# Create temp table
make_temp_table2(cur, conn)
check_db_content(cur)

# Check new test_table
cur.execute('''
			SELECT column_name, data_type 
			FROM information_schema.columns 
			WHERE table_name = 'test_table';
		    ''')
cur.fetchall()

cur.execute('''
			SELECT * FROM test_table
			LIMIT 1
		    ''')
cur.fetchall()


# update test_table (fill with data from recipes table)
cur.execute('''
			INSERT INTO test_table ("recipesID", "categories", "ingredients")
			SELECT "recipesID", "categories", "ingredients"
			FROM recipes;
			''')
conn.commit()

# show inserted data and overall size
cur.execute('''
			SELECT * FROM test_table
			LIMIT 10
		    ''')
cur.fetchall()

cur.execute(''' SELECT count(*) FROM test_table ''')
cur.fetchall()

cur.execute(''' SELECT count(*) FROM recipes ''')
cur.fetchall()

# Add tsvector column of "categories" and "ingredients" columns
cur.execute('''
			UPDATE test_table
			SET categories_tsv = to_tsvector(categories);
			
			UPDATE test_table
			SET ingredients_tsv = to_tsvector(ingredients);
			''')
conn.commit()

# show inserted data
cur.execute('''
			SELECT * FROM test_table
			LIMIT 10
		    ''')
cur.fetchall()


# Not sure how much use the ingredients column is actually going to be...


# drop test table
drop_temp_table(cur, conn)
check_db_content(cur)


# ===========================================================================
# Add tsvector column of "categories" to recipes
# Show current column names
check_table_columns(cur, 'recipes')

# Add new column: categories_tsv
cur.execute('''
			ALTER TABLE recipes
			ADD COLUMN categories_tsv TSVECTOR;
		    ''')
conn.commit()

# Check if new column is there
check_table_columns(cur, 'recipes')

# Populate categories_tsv
cur.execute('''
			UPDATE recipes
			SET categories_tsv = to_tsvector(categories);
			''')
conn.commit()

# show data
cur.execute('''
			SELECT * FROM recipes
			LIMIT 10
		    ''')
cur.fetchall()


# check output of websearch_to_tsquery()
cur.execute(sql.SQL(
            """
            SELECT websearch_to_tsquery('simple', '"vegan cookies"');
            """))
cur.fetchall()

# Test free search with tsquery
search_term = 'vegan cookies'
N = 10
cur.execute(sql.SQL(
            """
            SELECT "recipesID", "title", "categories_tsv",
                ts_rank_cd(categories_tsv, query) AS rank
            FROM public.recipes, websearch_to_tsquery('simple', %s) query
            WHERE query @@ categories_tsv
            ORDER BY rank ASC
            LIMIT %s
            """).format(), [search_term, N])
cur.fetchall()


# ===========================================================================
# Add ts_vector column for a combination of both title and categories, with
# an added weight of which column is more important
# See https://www.postgresql.org/docs/current/textsearch-controls.html

# Create temporary DB for testing
def make_temp_table3(cur, conn):
	cur.execute(
		'''
		CREATE TABLE public.test_table
		(
			"recipesID" bigint NOT NULL,
			categories VARCHAR,
			title VARCHAR,
			combined_tsv TSVECTOR,
			PRIMARY KEY ("recipesID")
		)
		
		WITH (
		    OIDS = FALSE
		)
		TABLESPACE pg_default;
		
		ALTER TABLE public.test_table
		OWNER to postgres;
		''')
	conn.commit()
	
	
# Create temp table
make_temp_table3(cur, conn)
check_db_content(cur)

# update test_table (fill with data from recipes table)
cur.execute('''
			INSERT INTO test_table ("recipesID", "categories", "title", "combined_tsv")
			SELECT "recipesID", "categories", "title", 
				setweight(to_tsvector(coalesce(title,'')), 'A') ||
				setweight(to_tsvector(coalesce(categories,'')), 'B') as tsv
			FROM recipes;
			''')
conn.commit()

# show inserted data and overall size
cur.execute('''
			SELECT * FROM test_table
			LIMIT 10
		    ''')
cur.fetchall()

# drop test table
drop_temp_table(cur, conn)
check_db_content(cur)


# ===========================================================================
# Add combined_tsv column to recipes table

cur.execute('''
			INSERT INTO test_table ("recipesID", "categories", "title", "combined_tsv")
			SELECT "recipesID", "categories", "title", 
				setweight(to_tsvector(coalesce(title,'')), 'A') ||
				setweight(to_tsvector(coalesce(categories,'')), 'B') as tsv
			FROM recipes;
			''')
conn.commit()

check_table_columns(cur, 'recipes')

# Add new column: categories_tsv
cur.execute('''
			ALTER TABLE recipes
			ADD COLUMN combined_tsv TSVECTOR;
		    ''')
conn.commit()

# Check if new column is there
check_table_columns(cur, 'recipes')

# Populate categories_tsv
cur.execute('''
			UPDATE recipes
			SET combined_tsv = 
				setweight(to_tsvector(coalesce(title,'')), 'A') ||
				setweight(to_tsvector(coalesce(categories,'')), 'B');
			''')
conn.commit()

# show data
cur.execute('''
			SELECT * FROM recipes
			LIMIT 10
		    ''')
cur.fetchall()


# check output of websearch_to_tsquery()
cur.execute(sql.SQL(
            """
            SELECT websearch_to_tsquery('simple', '"vegan cookies"');
            """))
cur.fetchall()

# Test free search with tsquery
search_term = 'vegan cookies'
N = 10
cur.execute(sql.SQL(
            """
            SELECT "recipesID", "title", "categories_tsv",
                ts_rank_cd(combined_tsv, query) AS rank
            FROM public.recipes, websearch_to_tsquery('simple', %s) query
            WHERE query @@ categories_tsv
            ORDER BY rank ASC
            LIMIT %s
            """).format(), [search_term, N])
cur.fetchall()


# TODO: Faciliate searching in categories column (using tsvector)
# For this, check makup of categories column in DB (if I actually included it)


# close DB connection
cur.close()


# eof









