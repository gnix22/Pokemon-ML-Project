import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score        
class PokeMachineClassifiers:
    def __init__(self, path:str):
        # load class csv
        self.df = pd.read_csv(path)
        self.df["combined_typing"] = self.df["Type 1"] + self.df["Type 2"].apply(lambda x: '/' + x if pd.notna(x) else '')
        self.df["norm_stats"] = (self.df["Base_Stats"] - min(self.df["Base_Stats"])) / (max(self.df["Base_Stats"]) - min(self.df["Base_Stats"]))
        # can exclude pretty much everything other than stats and typing
        # this is due to the fact that Base_Stats are total of individual stats
        keep_col_list = ["Name","Base_Stats",
                         "Type 1","Type 2","number_immune","number_not_effective","number_normal",
                         "number_super_effective","combined_typing"]
        self.df = self.df[keep_col_list]
    def decision_tree_classifier(self):
        df = self.df
        # for decision tree, set target as name, and based on
        # then predict the best team composition
        # this works by each iteration the model is retrained after the last type of the prediction is removed from the pool to choose from
        team = []
        pool = df.copy()
        for i in range(6):
            # train current set
            target = pool["Type 1"]
            features = pool.select_dtypes(include=[np.number])
            features_train, features_test, target_train, target_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
            decision_tree.fit(features_train, target_train)
            predictions = decision_tree.predict(features_test)
            print(f"Iteration {i+1} Accuracy: {accuracy_score(target_test, predictions):.2%}")
            # ideal stats that a pokemon would be expected to have to be strong.
            max_stats = max(pool["Base_Stats"])
            max_immune = max(pool["number_immune"])
            max_not_effective = max(pool["number_not_effective"])
            med_norm = pool["number_normal"].median()
            min_super_effective = min(pool["number_super_effective"])
            # predict with decision tree
            new_pokemon = pd.DataFrame(
                [[max_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
                columns=features.columns
            )
            predicted_type = decision_tree.predict(new_pokemon)[0]
            # find the actual predicted pokemon
            type_pool = pool[pool["Type 1"] == predicted_type]
            best_pokemon = type_pool.loc[type_pool["Base_Stats"].idxmax()]
            team.append(best_pokemon)
            print(f"Pick {i+1}: {best_pokemon['Name']} ({predicted_type})")
            # we selected the strongest of a type, now remove
            pool = pool[pool["Type 1"] != predicted_type]
        print("Final Team:")
        team_df = pd.DataFrame(team)[["Name", "combined_typing", "Base_Stats","number_immune",
                                      "number_not_effective","number_normal","number_super_effective"]]
        print(team_df)
    def knn_classifier(self):
        df = self.df
        pool = df.copy()
        team = []
        for i in range(6):
            # train on pool, just like decision tree
            target = pool["Type 1"]
            features = pool.select_dtypes(include=[np.number])
            features_train, features_test, target_train, target_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            k_values = range(1, 21)
            cv_scores = []
            for k in k_values:
                clf = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                        algorithm='brute', p=2)
                scores = cross_val_score(clf, features, target, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())
            best_k = k_values[np.argmax(cv_scores)]
            print(f"\nBest k: {best_k}")
            knn = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', algorithm='brute', p=2)
            knn.fit(features_train, target_train)
            predictions = knn.predict(features_test)
            print(f"Iteration {i+1} Accuracy: {accuracy_score(target_test, predictions):.2%}")
            max_stats = max(pool["Base_Stats"])
            max_immune = max(pool["number_immune"])
            max_not_effective = max(pool["number_not_effective"])
            med_norm = pool["number_normal"].median()
            min_super_effective = min(pool["number_super_effective"])
            new_pokemon = pd.DataFrame(
                [[max_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
                columns=features.columns
            )
            predicted_type = knn.predict(new_pokemon)[0]
            type_pool = pool[pool["Type 1"] == predicted_type]
            best_pokemon = type_pool.loc[type_pool["Base_Stats"].idxmax()]
            team.append(best_pokemon)
            print(f"Pick {i+1}: {best_pokemon['Name']} ({predicted_type})")
            # remove selected type from pool
            pool = pool[pool["Type 1"] != predicted_type]
        # move final team print outside the loop
        print("Final Team:")
        team_df = pd.DataFrame(team)[["Name", "combined_typing", "Base_Stats", "number_immune",
                                    "number_not_effective", "number_normal", "number_super_effective"]]
        print(team_df)
class UnsupervisedLearners:
    