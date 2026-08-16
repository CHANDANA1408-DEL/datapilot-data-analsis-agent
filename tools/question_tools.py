import pandas as pd

from tools.data_analysis import (
    calculate_total,
    calculate_average,
    calculate_minimum,
    calculate_maximum,
    group_and_sum,
    filter_data,
    rank_by_column,
    quarterly_sum,
    calculate_growth,
)


def get_total(
    df: pd.DataFrame,
    column: str
) -> float:
    """Calculate the total of a numeric column."""

    return calculate_total(df, column)


def get_average(
    df: pd.DataFrame,
    column: str
) -> float:
    """Calculate the average of a numeric column."""

    return calculate_average(df, column)


def get_minimum(
    df: pd.DataFrame,
    column: str
) -> float:
    """Find the minimum value of a numeric column."""

    return calculate_minimum(df, column)


def get_maximum(
    df: pd.DataFrame,
    column: str
) -> float:
    """Find the maximum value of a numeric column."""

    return calculate_maximum(df, column)


def get_grouped_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> list[dict]:
    """Group by a column and calculate the sum."""

    result = group_and_sum(
        df,
        group_column,
        value_column
    )

    return result.to_dict(orient="records")


def get_filtered_data(
    df: pd.DataFrame,
    column: str,
    value: str
) -> list[dict]:
    """Filter rows where a column equals a value."""

    result = filter_data(
        df,
        column,
        value
    )

    return result.to_dict(orient="records")


def get_top_values(
    df: pd.DataFrame,
    column: str,
    top_n: int = 5
) -> list[dict]:
    """Return the top rows based on a numeric column."""

    result = rank_by_column(
        df,
        column,
        ascending=False,
        top_n=top_n
    )

    return result.to_dict(orient="records")


def get_quarterly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str
) -> list[dict]:
    """Calculate a numeric value by quarter."""

    result = quarterly_sum(
        df,
        date_column,
        value_column
    )

    return result.to_dict(orient="records")