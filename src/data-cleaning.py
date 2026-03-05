import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../data/pokemon_data.csv")

## Note that one hot likely best for prediction of regression based questions since the dummies 
## would be hard to specifically predict? Maybe not though, need to look through old files.
# get hot one encoding of dual type to use in corr matrix
df["combined_typing"] = df["Type 1"] + df["Type 2"].apply(lambda x: '/' + x if pd.notna(x) else '')
one_hot_encoded_combined = pd.get_dummies(df["combined_typing"],dtype='uint8')
# get number of each type
sums = one_hot_encoded_combined.sum().to_list()
list_encoded = one_hot_encoded_combined.columns.to_list()

# can exclude pretty much everything other than stats and typing
# this is due to the fact that Base_Stats are total of individual stats
keep_col_list = ["Name","Base_Stats","Type 1","Type 2","number_immune","number_not_effective","number_normal","number_super_effective","combined_typing"]
selected_df = df[keep_col_list]
print(selected_df.info())


only_numeric = selected_df.select_dtypes(include=[np.number])
correlation = only_numeric.corr()
print("correlation matrix: ")
print(correlation)
