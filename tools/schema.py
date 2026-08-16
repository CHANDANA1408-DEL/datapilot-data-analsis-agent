import pandas as pd


def get_dataset_schema(df: pd.DataFrame) -> dict:
    """
    Return structural information about the dataset.
    """

    columns = []

    for column in df.columns:

        columns.append(
            {
                "name": column,
                "data_type": str(df[column].dtype),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique()),
            }
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }