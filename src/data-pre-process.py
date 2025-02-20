import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 42
train_size, val_size, test_size = 0.6, 0.2, 0.2

df = pd.read_csv('../dataset/encoded_data.csv')
normalized = (df - df.min()) / (df.max() - df.min())

common = normalized[normalized['HadHeartAttack'] == 0]
rare = normalized[normalized['HadHeartAttack'] == 1]


def split_data(data, train_size=0.6, val_size=0.2, test_size=0.2):
    train_val, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size / (train_size + val_size), random_state=random_state)
    return train, val, test


train_common, val_common, test_common = split_data(common, train_size, val_size, test_size)
train_rare, val_rare, test_rare = split_data(rare, train_size, val_size, test_size)

train = pd.concat([train_common, train_rare])
val = pd.concat([val_common, val_rare])
test = pd.concat([test_common, test_rare])

train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)

train.to_csv('../dataset/train.csv', index=False)
val.to_csv('../dataset/val.csv', index=False)
test.to_csv('../dataset/test.csv', index=False)