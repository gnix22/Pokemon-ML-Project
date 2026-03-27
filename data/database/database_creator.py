import pandas as pd
import sqlite3
import re # regex
import os
# using the advantage of sql relational database management system approach to make certain predictors
# also using sqlite3 compatibility to my advantage with python due to huge numbers of predictors already
# in order to create tables in an easy way (that is, slightly automated, and not manually due to the number
# of predictors [not sure if this is typical in an applied setting, but it seemed the most natural approach
# given my knowledge]).

# configuration constants of database filenames and sql files
DATABASE = "poke-data.db"
DATA_DIR = "../"
SCHEME_SCHEME = "schema.sql"
# some mappings
tables = {
    #"pokemon" : f"{DATA_DIR}pokemon_data.csv",
    "moves": f"{DATA_DIR}moves.csv",
    #"poke-moves": f"{DATA_DIR}pokemon_moves.csv"
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
# create table schema using above for each table
def generate_table_schema(table):
    for table, table_path in tables.items():
        df = pd.read_csv(table_path, nrows=5) # unnecessary to load entire dataframe for each path
        df.columns = [sql_column_clean_up(column) for column in df.columns]
        column_names = []
        for col, dtype in zip(df.columns, df.dtypes):
            sql_type = dtype_to_sql(dtype=dtype)
            # maybe figure out a primary key determination? not sure if I want that specifically yet
            column_names.append(f'"{col}" {sql_type}') # set the column name and type for typical convention of CREATE TABLE
        column_string = ",\n".join(column_names) # puts vertical
        schema_generation = f'CREATE TABLE IF NOT EXISTS "{table}" (\n{column_string}\n);\n' 
    # open the database file
    with open(SCHEME_SCHEME, "w") as file:
        file.write(schema_generation)
    print("schema tables generated successfully")
# load respective csvs and insert
def data_loading(tables):
    connection = sqlite3.connect(DATABASE)
    # apply the schema
    with open(SCHEME_SCHEME) as file:
        connection.executescript(file.read())
    for table, table_path in tables.items():
        df = pd.read_csv(table_path)
        df.columns = [sql_column_clean_up(column) for column in df.columns]
        df.to_sql(table, connection, if_exists="replace", index=False)
        print(f"successful loading of {len(df)} rows into '{table}'")
    # close connection
    connection.close()

# run it
generate_table_schema(tables)
data_loading(tables)