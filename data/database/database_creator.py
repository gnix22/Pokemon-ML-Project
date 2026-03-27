import pandas as pd
import sqlite3
import re # regex
import os
# using the advantage of sql relational database management system approach to make certain predictors
# also using sqlite3 compatibility to my advantage with python due to huge numbers of predictors already
# in order to create tables in an easy way.

# configuration constants of database filenames and sql files
DATABASE = "poke-data.db"
DATA_DIR = "../"
SCHEME_SCHEME = "schema.sql"

# some mappings
tables = {
    "pokemon" : f"{DATA_DIR}pokemon_data.csv",
    "moves": f"{DATA_DIR}moves.csv",
    "poke-moves": f"{DATA_DIR}pokemon_moves.csv"
}

# try and infer what df type is to convert to sql
def dtype_to_sql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    else:
        return "TEXT"

# clean the column names into a "sql" presentable format using regex
def sql_column_clean_up(name):
    name = re.sub(r'[^a-zA-Z0-9_]', "_", name.strip()) # substitute header with str
    name = re.sub(r'_+', '_', name).strip('_').lower()
    return name

# need to figure out create table schema using above for each table, 