import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# %% md
# 相关性分析
# %%
df = pd.read_csv('../dataset/encoded_data.csv')
data = pd.read_csv('../dataset/encoded_data.csv')


# %%
# 分类函数
def classify_columns(df):
    real_numbers = []
    natural_numbers = []
    binary_classification = []

    for column in df.columns:
        if df[column].unique().shape[0] == 2:
            binary_classification.append(column)
        elif (df[column] % 1 == 0).all():
            natural_numbers.append(column)
        else:
            real_numbers.append(column)

    return real_numbers, natural_numbers, binary_classification


# 调用分类函数
real_numbers, natural_numbers, binary_classification = classify_columns(df)

print("实数列：", real_numbers)
print("自然数列：", natural_numbers)
print("二值分类列：", binary_classification)
# %% md
## 二值分类列
# %%
binary_data = data[binary_classification]

# 计算每个属性与 HadHeartAttack 的概率
# 计算联合概率 P(A & HadHeartAttack)
joint_prob = binary_data.copy()
joint_prob['HadHeartAttack'] = data['HadHeartAttack']  # 添加 HadHeartAttack 列用于联合计算

# 计算每一列的置信度和提升度
confidence_dict = {}
lift_dict = {}

for column in binary_data.columns:
    # 计算 P(A & HadHeartAttack)
    joint_count = len(joint_prob[(joint_prob[column] == 1) & (joint_prob['HadHeartAttack'] == 1)])
    # 计算 P(A)
    prob_A = binary_data[column].sum() / len(binary_data)
    # 计算 P(HadHeartAttack)
    prob_HadHeartAttack = data['HadHeartAttack'].sum() / len(data)
    # 计算 P(A | HadHeartAttack)
    prob_A_given_HadHeartAttack = joint_count / data['HadHeartAttack'].sum()

    # 置信度 = P(A | HadHeartAttack)
    confidence = prob_A_given_HadHeartAttack
    confidence_dict[column] = confidence

    # 提升度 = P(A & HadHeartAttack) / (P(A) * P(HadHeartAttack))
    lift = joint_count / (prob_A * prob_HadHeartAttack * len(binary_data))
    lift_dict[column] = lift

# 创建结果的 DataFrame
result_df = pd.DataFrame({
    'Column': binary_data.columns,
    'Confidence': confidence_dict.values(),
    'Lift': lift_dict.values(),
})

# 输出结果
result_df.sort_values(
    by='Lift', ascending=False, ignore_index=True, inplace=True
)
result_df.to_csv(
    '../asserts/binary_classification_correlation.csv', index=False
)

# %% md
## 自然数列
# %%
natural_data = data[natural_numbers]
target = data['HadHeartAttack']  # 二值属性


# 定义计算 Cramér's V 的函数
def cramers_v(chi2, n, k, r):
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))


# 初始化字典来存储卡方检验和 Cramér's V 结果
chi2_dict = {}
p_value_dict = {}
cramers_v_dict = {}

# 对每一列进行卡方检验和 Cramér's V 计算
for column in natural_data.columns:
    # 计算卡方检验的列联表
    contingency_table = pd.crosstab(natural_data[column], target)

    # 计算卡方检验结果
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # 计算 Cramér's V
    n = contingency_table.sum().sum()  # 总样本数
    k = contingency_table.shape[0]  # 行数
    r = contingency_table.shape[1]  # 列数
    cramers_v_value = cramers_v(chi2, n, k, r)

    # 保存结果
    chi2_dict[column] = chi2
    p_value_dict[column] = p
    cramers_v_dict[column] = cramers_v_value

# 将结果汇总到一个 DataFrame 中
results_df = pd.DataFrame({
    'Column': natural_data.columns,
    'Chi2 Statistic': chi2_dict.values(),
    'P-value': p_value_dict.values(),
    'Cramér\'s V': cramers_v_dict.values()
})

# 显示结果
results_df.to_csv(
    '../asserts/natural_numbers_correlation.csv', index=False
)
# %% md
## 实数列
# %%
real_data = data[real_numbers]
target = data['HadHeartAttack']  # 二值属性

# 存储结果的字典
correlation_dict = {}
p_value_dict = {}

# 对每个连续属性列进行点双列相关系数计算
for column in real_data.columns:
    # 计算点双列相关系数（Point-Biserial Correlation）
    corr, p_value = pointbiserialr(real_data[column], target)
    correlation_dict[column] = corr
    p_value_dict[column] = p_value

# 汇总结果到 DataFrame 中
results_df = pd.DataFrame({
    'Column': real_data.columns,
    'Point-Biserial Correlation': correlation_dict.values(),
    'P-value': p_value_dict.values()
})
results_df.to_csv(
    '../asserts/real_numbers_correlation.csv', index=False
)