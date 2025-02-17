import lightgbm as lgb
import pandas as pd

# 读取数据
df = pd.read_csv('../dataset/train.csv')
X = df.drop(columns='HadHeartAttack')
y = df['HadHeartAttack']

# 假设 X 是特征数据，y 是目标变量
train_data = lgb.Dataset(X, label=y)
params = {'objective': 'binary', 'metric': 'binary_error'}

# 训练模型
model = lgb.train(params, train_data, 100)

# 获取特征的重要性
importances = model.feature_importance()

# 根据重要性排序特征
indices = importances.argsort()[::-1]
X_new = X.iloc[:, indices[:5]]  # 选择前 5 个重要特征

print(X_new.columns)  # 输出前 5 个重要特征的名称