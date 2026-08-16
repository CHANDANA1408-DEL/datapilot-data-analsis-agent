import pandas as pd


def analyze_dataset(
    df: pd.DataFrame,
    operation: str,
    column: str | None = None,
    group_by: str | None = None,
    aggregation: str = "sum",
    filters: dict | None = None,
    top_n: int | None = None,
) -> object:
    """
    Perform a generic deterministic analysis on a DataFrame.

    Supported operations:

    - aggregate
    - group
    - filter
    - filter_aggregate
    - rank
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    operation = operation.lower().strip()
    aggregation = aggregation.lower().strip()

    if column is not None and column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if group_by is not None and group_by not in df.columns:
        raise ValueError(
            f"Column '{group_by}' does not exist."
        )

    if operation == "aggregate":

        if column is None:
            raise ValueError(
                "column is required for aggregate operation."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        return _aggregate(
            df[column],
            aggregation,
        )

    if operation == "group":

        if group_by is None:
            raise ValueError(
                "group_by is required for group operation."
            )

        if column is None:
            raise ValueError(
                "column is required for group operation."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        result = (
            df.groupby(group_by)[column]
            .agg(aggregation)
            .reset_index()
        )

        return result.sort_values(
            by=column,
            ascending=False,
        ).reset_index(drop=True)

    if operation == "filter":

        if filters is None:
            raise ValueError(
                "filters are required for filter operation."
            )

        result = df.copy()

        for filter_column, filter_value in filters.items():

            if filter_column not in result.columns:
                raise ValueError(
                    f"Column '{filter_column}' does not exist."
                )

            result = result[
                result[filter_column] == filter_value
            ]

        return result.reset_index(drop=True)

    if operation == "filter_aggregate":

        if filters is None:
            raise ValueError(
                "filters are required for filter_aggregate."
            )

        if column is None:
            raise ValueError(
                "column is required for filter_aggregate."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        filtered = df.copy()

        for filter_column, filter_value in filters.items():

            if filter_column not in filtered.columns:
                raise ValueError(
                    f"Column '{filter_column}' does not exist."
                )

            filtered = filtered[
                filtered[filter_column] == filter_value
            ]

        return _aggregate(
            filtered[column],
            aggregation,
        )

    if operation == "rank":

        if column is None:
            raise ValueError(
                "column is required for rank operation."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        if top_n is None:
            top_n = 5

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        return (
            df.sort_values(
                by=column,
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


def _aggregate(
    series: pd.Series,
    aggregation: str,
) -> float:

    if aggregation == "sum":
        return float(series.sum())

    if aggregation == "mean":
        return float(series.mean())

    if aggregation == "min":
        return float(series.min())

    if aggregation == "max":
        return float(series.max())

    if aggregation == "median":
        return float(series.median())

    if aggregation == "count":
        return int(series.count())

    if aggregation == "std":
        return float(series.std())

    raise ValueError(
        f"Unsupported aggregation: {aggregation}"
    )