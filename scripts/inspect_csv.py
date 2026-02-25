import pandas as pd

raw = pd.read_csv("data/processed/XAUUSD_1H.csv")

df = raw.iloc[:, 0].str.split("\t", expand=True)

print("Number of rows:", len(df))
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())
