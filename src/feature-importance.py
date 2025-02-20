import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../asserts/feature_importance.csv')
df.set_index('Unnamed: 0', inplace=True)
df.plot(kind='bar', figsize=(10, 6), width=0.8)
plt.title('Feature Importance Comparison', fontsize=16)
plt.tight_layout()
plt.xlabel('', fontsize=14)
plt.savefig('../asserts/feature_importance.png')
plt.show()
