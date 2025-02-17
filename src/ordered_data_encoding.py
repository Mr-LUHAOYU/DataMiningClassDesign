r"""Convert ordered categorical text data to natural number encoding."""
import pandas as pd

df = pd.read_csv('../dataset/Heart Disease Dataset.csv')
df.drop(columns=['State'], inplace=True)

text2number = open('../constants/map.txt', 'r').read().strip()
text2number = eval(text2number)
text2number = {key: {v: i for i, v in enumerate(value)} for key, value in text2number.items()}
for col in df.columns:
    if col in text2number:
        df[col] = df[col].apply(lambda x: text2number[col][x])
    elif len(df[col].unique()) == 2:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

df.to_csv('../dataset/encoded_data.csv', index=False)