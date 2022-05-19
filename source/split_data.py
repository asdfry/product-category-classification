import pandas as pd
from pandas.io.parsers import read_csv
from tqdm import tqdm

df_train = pd.DataFrame()
df_valid = pd.DataFrame()
df_test = pd.DataFrame()

prefix = "d3_"
if prefix:
    csv_file = f"../data/{prefix}_goods_nori_drop.csv"
else:
    csv_file = "../data/goods_nori_drop.csv"
print(f"> reading csv to dataframe... [{csv_file}]")

df = read_csv(csv_file)
category_lst = list(df["category_id"].unique())

for idx, category_id in enumerate(tqdm(category_lst), start=1):
    sample_train = df[df["category_id"] == category_id]
    sample_valid = sample_train.sample(frac=0.2)
    sample_train = sample_train.drop(sample_valid.index)
    sample_test = sample_valid.sample(frac=0.5)
    sample_valid = sample_valid.drop(sample_test.index)

    df_train = pd.concat([df_train, sample_train])
    df_valid = pd.concat([df_valid, sample_valid])
    df_test = pd.concat([df_test, sample_test])

    del [sample_train, sample_valid, sample_test]

del [df]

if prefix:
    path = f"../data/splited_data/{prefix}_"
else:
    path = "../data/splited_data/"
df_lst = [(df_train, "train"), (df_valid, "valid"), (df_test, "test")]

for df in df_lst:
    df_name = df[1]
    # 데이터프레임에 nan이 존재하는 경우 drop
    if df[0].isna().values.any():
        print(f"> df_{df_name} drop nan...")
        df[0].dropna(axis=0, inplace=True)
    df[0].to_csv(f"{path + df_name}.csv", index=False)
    print(f"> saved at {path + df_name}.csv (length={len(df[0])}, exist_nan={df[0].isna().values.any()})")
