import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# load csv
df = pd.read_csv("../data/pokemon_data.csv")
########################################################### TYPE CLASSIFICATION ########################################################################
df["combined_typing"] = df["Type 1"] + df["Type 2"].apply(lambda x: '/' + x if pd.notna(x) else '')
df["total_effective"] = df["number_immune"] + df["number_not_effective"] + df["number_normal"] + df["number_super_effective"] # possibly better for just minimizing the number of
                                                                                                                              # moves effective against the pokemon
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
    # using a regression model to create a multi target like model to better predict the best typings.
    linear_features = pool.select_dtypes(include=[np.number]).drop(columns=["Base_Stats"])
    linear_target = pool["Base_Stats"]
    linear_regressor = LinearRegression().fit(linear_features,linear_target) # train on whole pool

    linear_pred = linear_regressor.predict(linear_features)
    mse = mean_squared_error(linear_target, linear_pred)
    r2 = r2_score(linear_target, linear_pred)
    print(f"regression mean squared: {mse:.2f}, R²: {r2:.4f}")

    # train current set
    target = pool["combined_typing"]
    features = pool.select_dtypes(include=[np.number])
    class_features_train, class_features_test, class_target_train, class_target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    # decision tree using prediction from regressor
    decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
    decision_tree.fit(class_features_train, class_target_train)
    # ideal stats that a pokemon would be expected to have to be strong.
    max_stats = max(pool["Base_Stats"])
    max_immune = max(pool["number_immune"])
    max_not_effective = max(pool["number_not_effective"])
    med_norm = pool["number_normal"].median()
    min_super_effective = min(pool["number_super_effective"])
    # dataframe for regressor
    ideal_input = pd.DataFrame(
        [[max_immune, max_not_effective, med_norm, min_super_effective]],
        columns=linear_features.columns
    )
    # use regressor to predict what Base_Stats this ideal pokemon would have
    predicted_base_stats = linear_regressor.predict(ideal_input)[0]
    # now predict with decision tree
    new_pokemon = pd.DataFrame(
        [[predicted_base_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
        columns=features.columns
    )
    predictions = decision_tree.predict(class_features_test)
    print(f"Iteration {i+1} Accuracy: {accuracy_score(class_target_test, predictions):.2%}")

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
