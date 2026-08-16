import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a profile of the dataset.

    The profile contains:
    - number of rows
    - number of columns
    - column names
    - data types
    - missing values
    """

    column_info = {}

    for column in df.columns:
        column_info[column] = {
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
        }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "columns_info": column_info,
    }