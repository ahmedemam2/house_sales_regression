import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def check_missing_values(df):
    missing_data = df.isnull()
    for column in missing_data.columns.values.tolist():
        print(column)
        print(missing_data[column].value_counts())
        print("")

# def clean_data(df):
#     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
#     # df = df.replace("?", np.NaN, inplace=True)
#     df = df.replace('', np.NaN, inplace=True)
#     return df

def validate_NaN(df):
    has_null = df.isnull().any()
    for column in has_null[has_null].index:
        null_indices = df.index[df[column].isnull()]
        print(f"Column: {column}")
        print(f"NaN indices: {null_indices.tolist()}")
        print("")


def replace_missing_values(df):
    has_null = df.isnull().any()
    for column in has_null[has_null].index:
        if df[column].dtype == 'int64' or df[column].dtype == 'float':
            df[column] = df[column].astype(float)
            df[column].replace(np.NaN, df[column].astype('float').mean(axis=0), inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def visualize(df):
    sns.regplot(x = "sqft_living", y = "price", data=df)
    plt.ylim(0,)
    plt.show()

def main():
    df = pd.read_csv("kc_house_High_Corr.csv")
    check_missing_values(df)
    validate_NaN(df)
    df = replace_missing_values(df)
    df.to_csv("kc_house_cleaned.csv")
    validate_NaN(df)
    visualize(df)
main()

