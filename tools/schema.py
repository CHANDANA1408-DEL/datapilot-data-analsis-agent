import pandas as pd


def infer_semantic_type(series: pd.Series) -> str:
    """
    Infer the semantic type of a dataset column.
    """

    dtype = series.dtype
    unique_count = series.nunique(dropna=True)

    column_name = str(series.name).lower()

    # ---------------------------------------------------------
    # ALREADY DATETIME
    # ---------------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # ---------------------------------------------------------
    # DATETIME DETECTION BY COLUMN NAME
    # ---------------------------------------------------------

    datetime_keywords = [
        "date",
        "time",
        "timestamp",
        "datetime",
    ]

    if any(
        keyword in column_name
        for keyword in datetime_keywords
    ):
        converted = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )

        if converted.notna().mean() >= 0.8:
            return "datetime"

    # ---------------------------------------------------------
    # IDENTIFIER DETECTION
    # ---------------------------------------------------------

    identifier_keywords = [
        "id",
        "number",
        "code",
    ]

    if any(
        keyword in column_name
        for keyword in identifier_keywords
    ):
        if unique_count == len(series):
            return "identifier"

    # ---------------------------------------------------------
    # STRING COLUMNS
    # ---------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):

        converted = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )

        if converted.notna().mean() >= 0.8:
            return "datetime"

        return "categorical"

    # ---------------------------------------------------------
    # BINARY CATEGORICAL
    # ---------------------------------------------------------

    if unique_count == 2:
        return "categorical"

    # ---------------------------------------------------------
    # NUMERIC
    # ---------------------------------------------------------

    if pd.api.types.is_numeric_dtype(dtype):

        if unique_count <= 10:
            return "ordinal"

        return "numeric"

    return "unknown"


def get_dataset_schema(
    df: pd.DataFrame,
) -> dict:
    """
    Return structural and semantic information
    about the dataset.
    """

    columns = []

    for column in df.columns:

        series = df[column]

        columns.append(
            {
                "name": column,
                "data_type": str(series.dtype),
                "semantic_type": infer_semantic_type(
                    series
                ),
                "missing_values": int(
                    series.isna().sum()
                ),
                "unique_values": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }