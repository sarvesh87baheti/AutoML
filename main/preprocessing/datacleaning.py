import pandas as pd
import numpy as np


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by performing:
      1. Drop unnamed columns (common in Excel/CSV exports)
      2. Fix inconsistent column types using majority type inference
      3. Remove completely NULL columns
      4. Handle missing values with median/mode imputation
      5. Remove duplicate rows
      6. Remove numeric outliers using 1.5*IQR rule
      7. Reset index after cleaning

    Returns a cleaned, warning-free DataFrame.
    """

    # ✅ Always work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1️⃣ Drop unnamed columns (like "Unnamed: 0" from CSVs)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # 2️⃣ Fix inconsistent column types
    for col in df.columns:
        non_null_values = df[col].dropna()
        if non_null_values.empty:
            continue

        # Find the most frequent data type in column
        majority_type = non_null_values.map(type).value_counts().idxmax()

        def convert_value(val):
            try:
                return majority_type(val)
            except Exception:
                return np.nan

        # Use .loc to safely assign values
        df.loc[:, col] = df[col].apply(convert_value)

    # 3️⃣ Drop columns that are completely NULL
    df.dropna(axis=1, how="all", inplace=True)

    # 4️⃣ Handle missing values (median for numeric, mode for categorical)
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            median_value = df[col].median()
            df.loc[:, col] = df[col].fillna(median_value)
        else:
            if not df[col].mode().empty:
                mode_value = df[col].mode().iloc[0]
                df.loc[:, col] = df[col].fillna(mode_value)
            else:
                df.loc[:, col] = df[col].fillna("")

    # 5️⃣ Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 6️⃣ Remove outliers using IQR for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].nunique() < 5:  # skip small unique sets (e.g., categories)
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Use .loc to filter safely
        df = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # 7️⃣ Reset index
    df.reset_index(drop=True, inplace=True)

    return df
