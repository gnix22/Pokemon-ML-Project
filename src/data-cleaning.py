import pandas as pd
import numpy as np

df = pd.read_csv("../data/pokemon_data.csv")
print(df.head(5))
print(df.columns)
# can exclude pretty much everything other than stats and typing
keep_col_list = ["Name","Attack","Defense","Sp. Attack","Sp. Defense","Speed","Base_Stats","Type 1","Type 2"]
selected_df = df[keep_col_list]
print(selected_df.head(5))
