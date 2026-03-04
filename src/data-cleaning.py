import pandas as pd
import numpy as np

df = pd.read_csv("../data/pokemon_data.csv")
# get hot one encoding of dual type to use in corr matrix
df["combined_typing"] = df["Type 1"] + '/' + df["Type 2"]
one_hot_encoded_combined = pd.get_dummies(df["combined_typing"],dtype='uint8')
list_encoded = one_hot_encoded_combined.columns.to_list()
df.drop('combined_typing', axis=1)
df = df.join(one_hot_encoded_combined)

# can exclude pretty much everything other than stats and typing
# this is due to the fact that Base_Stats are total of individual stats
keep_col_list = ["Name","Base_Stats","Type 1","Type 2","number_immune","number_not_effective","number_normal","number_super_effective"]
keep_col_list.extend(list_encoded)
selected_df = df[keep_col_list]
only_numeric = selected_df.select_dtypes(include=[np.number])

correlation = only_numeric.corr()
print("correlation matrix: ")
print(correlation)
