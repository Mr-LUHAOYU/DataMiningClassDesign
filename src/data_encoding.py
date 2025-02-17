r"""Convert text data to natural number encoding."""
import pandas as pd

df = pd.read_csv('../dataset/Heart Disease Dataset.csv')
df.drop(columns=['State'], inplace=True)

exec(open('../constants/map.txt', 'r').read().strip())

for col in df.columns:
    if col in text2number:
        df[col] = df[col].apply(lambda x: text2number[col][x])
    elif col == 'RaceEthnicityCategory':
        df[col] = df[col].apply(lambda x: RaceEthnicityCategory_dict[x])
    elif len(df[col].unique()) == 2:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

df.to_csv('../dataset/encoded_data.csv', index=False)