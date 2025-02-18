import pandas as pd

train_data = pd.read_csv("../dataset/train.csv")
valid_data = pd.read_csv("../dataset/val.csv")
test_data = pd.read_csv("../dataset/test.csv")

X_train = train_data.drop("HadHeartAttack", axis=1)
y_train = train_data["HadHeartAttack"]
X_valid = valid_data.drop("HadHeartAttack", axis=1)
y_valid = valid_data["HadHeartAttack"]
X_test = test_data.drop("HadHeartAttack", axis=1)
y_test = test_data["HadHeartAttack"]