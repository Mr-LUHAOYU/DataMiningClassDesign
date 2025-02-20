import pandas as pd
from sklearn.model_selection import train_test_split

train_size, val_size, test_size = 0.6, 0.2, 0.2

df = pd.read_csv('../dataset/encoded_data.csv')
normalized = (df - df.min()) / (df.max() - df.min())

common = normalized[normalized['HadHeartAttack'] == 0]
rare = normalized[normalized['HadHeartAttack'] == 1]


def split_data(data, train_size=0.6, val_size=0.2, test_size=0.2):
    train_val, test = train_test_split(data, test_size=test_size)
    train, val = train_test_split(train_val, test_size=val_size / (train_size + val_size))
    return train, val, test


train_common, val_common, test_common = split_data(common, train_size, val_size, test_size)
train_rare, val_rare, test_rare = split_data(rare, train_size, val_size, test_size)

train = pd.concat([train_common, train_rare])
val = pd.concat([val_common, val_rare])
test = pd.concat([test_common, test_rare])

train = train.sample(frac=1).reset_index(drop=True)
val = val.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

target_dir = '../dataset/data0'
import os
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
train.to_csv(f'{target_dir}/train.csv', index=False)
val.to_csv(f'{target_dir}/val.csv', index=False)
test.to_csv(f'{target_dir}/test.csv', index=False)