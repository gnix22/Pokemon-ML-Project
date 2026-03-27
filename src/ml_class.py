import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score   

# It should be noted that these classifiers don't particularly work that well given just stats and numbers of efficacy in blocking moves. This is likely due to the fact
# that largely types themselves are relatively balanced across all pokemon games. Working to test if creation of predictors are useful.
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
    ## decision tree function
    def decision_tree_classifier(self):
        print(".........decision classifier.........")
        df = self.df
        # rework to just predict strongest typing
        # train current set
        target = df["Type 1"]
        features = df.select_dtypes(include=[np.number])
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        # define decision tree
        decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
        decision_tree.fit(features_train, target_train)
        predictions = decision_tree.predict(features_test)
        print(f"Accuracy: {accuracy_score(target_test, predictions):.2%}")
        # ideal stats that a pokemon would be expected to have to be strong.
        max_stats = max(df["Base_Stats"])
        max_immune = max(df["number_immune"])
        max_not_effective = max(df["number_not_effective"])
        med_norm = df["number_normal"].median()
        min_super_effective = min(df["number_super_effective"])
        # predict with decision tree
        new_pokemon = pd.DataFrame(
            [[max_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
            columns=features.columns
        )
        predicted_type = decision_tree.predict(new_pokemon)[0]
        # find the actual predicted pokemon
        type_pred = df[df["Type 1"] == predicted_type]
        best_pokemon = type_pred.loc[type_pred["Base_Stats"].idxmax()]
        print(best_pokemon)
    ## set up the knn classifier
    def knn_classifier(self):
        print(".........knn classifier.........")
        df = self.df
        # train on pool, just like decision tree
        target = df["Type 1"]
        features = df.select_dtypes(include=[np.number])
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
        print(f"Accuracy: {accuracy_score(target_test, predictions):.2%}")
        max_stats = max(df["Base_Stats"])
        max_immune = max(df["number_immune"])
        max_not_effective = max(df["number_not_effective"])
        med_norm = df["number_normal"].median()
        min_super_effective = min(df["number_super_effective"])
        new_pokemon = pd.DataFrame(
            [[max_stats, max_immune, max_not_effective, med_norm, min_super_effective]],
            columns=features.columns
        )
        predicted_type = knn.predict(new_pokemon)[0]
        type_pred = df[df["Type 1"] == predicted_type]
        best_pokemon = type_pred.loc[type_pred["Base_Stats"].idxmax()]
        print(best_pokemon)
# due to there not being much insight given from the above models, it may be potentially beneficial to create
# labels that are found via unsupervised learning methods. 
class UnsupervisedLearners:
    def __init__(self,csv):
        self.df = pd.read_csv()

