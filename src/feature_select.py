import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

pd.options.display.float_format = '{:.4f}'.format

df = pd.read_csv('../dataset/train.csv')
X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

model = RandomForestClassifier()
model.fit(X, y)
perm_importance = permutation_importance(model, X, y)

feature_importances = model.feature_importances_
importances_mean = perm_importance.importances_mean

columns = X.columns
feature_importances = pd.DataFrame(feature_importances, index=columns)
importances_mean = pd.DataFrame(importances_mean, index=columns)
corr = pd.concat([feature_importances, importances_mean], axis=1)
corr.columns = ['feature_importances', 'perm_importance']
corr.to_csv('../asserts/feature_importance.csv', index=True)
