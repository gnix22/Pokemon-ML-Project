import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class PokeMachineClassifiers:
    def __init__(self, path:str):
        # load class csv
        self.df = pd.read_csv(path)
        self.df["combined_typing"] = self.df["Type 1"] + self.df["Type 2"].apply(lambda x: '/' + x if pd.notna(x) else '')
        # can exclude pretty much everything other than stats and typing
        # this is due to the fact that Base_Stats are total of individual stats
        keep_col_list = ["Name","Base_Stats","Type 1","Type 2","number_immune","number_not_effective","number_normal","number_super_effective","combined_typing"]
        self.df = self.df[keep_col_list]
    def type_classifier(self):
        df = self.df
        # for decision tree, set target as typing and use numeric features only
        # then predict the best team composition, where each pokemon has to be of a different typing.
        # this works by each iteration the model is retrained after the last type of the prediction is removed from the pool to choose from
        team = []
        pool = df.copy()
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
    def knn_classifier(self):
        df = self.df
        # train set
        target = df["combined_typing"]
        features = df.select_dtypes(include=[np.number])
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        k_values = range(1, 21)
        error_rates = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(features_train, target_train)
            target_pred = knn.predict(features_test)
            error_rate = 1 - accuracy_score(target_test, target_pred)
            error_rates.append(error_rate)
        min_err = min(error_rates)
        # set up predictions for team similar to tree classifier

