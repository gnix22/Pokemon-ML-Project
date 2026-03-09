from ml_class import PokeMachineClassifiers

csv = "../data/pokemon_data.csv"
model = PokeMachineClassifiers(csv)

model.decision_tree_classifier()
