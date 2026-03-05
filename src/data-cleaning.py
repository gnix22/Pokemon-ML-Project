import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load csv
df = pd.read_csv("../data/pokemon_data.csv")
########################################################## REGRESSION #################################################################################
## Note that one hot likely best for prediction of regression based questions since the dummies 
## would be hard to specifically predict? Maybe not though, need to look through old files.
df["combined_typing"] = df["Type 1"] + df["Type 2"].apply(lambda x: '/' + x if pd.notna(x) else '')
one_hot_encoded_combined = pd.get_dummies(df["combined_typing"],dtype='uint8')
# get number of each type
sums = one_hot_encoded_combined.sum().to_list()
list_encoded = one_hot_encoded_combined.columns.to_list()

########################################################### TYPE CLASSIFICATION ########################################################################
# can exclude pretty much everything other than stats and typing
# this is due to the fact that Base_Stats are total of individual stats
keep_col_list = ["Name","Base_Stats","Type 1","Type 2","number_immune","number_not_effective","number_normal","number_super_effective","combined_typing"]
selected_df = df[keep_col_list]
# for decision tree, set target as typing and use numeric features only
# then predict the best team composition, where each pokemon has to be of a different typing.
# this works by each iteration the model is retrained after the last type of the prediction is removed from the pool to choose from
team = []
pool = selected_df.copy()
for i in range(6):
    # train current set
    target = pool["combined_typing"]
    features = pool.select_dtypes(include=[np.number])
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
    decision_tree.fit(features_train, target_train)
    # ideal stats that a pokemon would be expected to have to be strong.
    max_stats = max(pool["Base_Stats"])
    max_immune = max(pool["number_immune"])
    max_not_effective = max(pool["number_not_effective"])
    med_norm = pool["number_normal"].median()
    min_super_effective = min(pool["number_super_effective"])
    new_pokemon = pd.DataFrame(
        [[max_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
        columns=features.columns
    )
    predictions = decision_tree.predict(features_test)
    print(f"Iteration {i+1} Accuracy: {accuracy_score(target_test, predictions):.2%}")

    predicted_type = decision_tree.predict(new_pokemon)[0]
    # find the actual predicted pokemon
    type_pool = pool[pool["combined_typing"] == predicted_type]
    best_pokemon = type_pool.loc[type_pool["Base_Stats"].idxmax()]
    team.append(best_pokemon)
    print(f"Pick {i+1}: {best_pokemon['Name']} ({predicted_type})")
    # we selected the strongest of a type, now remove
    pool = pool[pool["combined_typing"] != predicted_type]
print("Final Team:")
team_df = pd.DataFrame(team)[["Name", "combined_typing", "Base_Stats","number_immune","number_not_effective","number_normal","number_super_effective"]]
print(team_df)
