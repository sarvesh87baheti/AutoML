import sys
import os
sys.path.append(os.path.abspath(".."))
## from Zip_Extracter.zipextracter import extract_zip_to_dataframe
from EDA import perform_eda, plot_correlation_heatmap, pca_reduction


import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame:
      - Fix inconsistent types (majority type per column)
      - Remove NULL columns
      - Handle missing values
      - Remove duplicates
      - Remove outliers using 1.5*IQR
    """
    # 1️⃣ Fix inconsistent column types
     # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for col in df.columns:
        types = df[col].dropna().map(type).value_counts()
        if not types.empty:
            majority_type = types.idxmax()

            def convert_value(val):
                try:
                    return majority_type(val)
                except Exception:
                    return np.nan

            df[col] = df[col].apply(convert_value)

    # 2️⃣ Remove completely NULL columns
    df.dropna(axis=1, how="all", inplace=True)

    # 3️⃣ Handle missing values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
          df[col] = df[col].fillna(df[col].median())
        else:
          df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 4️⃣ Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 5️⃣ Remove outliers using 1.5*IQR
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    df.reset_index(drop=True, inplace=True)
    return df