import pandas as pd


def calculate_total(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the total of a numeric column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric.")

    return float(df[column].sum())


def calculate_average(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the average of a numeric column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric.")

    return float(df[column].mean())


def calculate_minimum(df: pd.DataFrame, column: str) -> float:
    """
    Find the minimum value in a numeric column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric.")

    return float(df[column].min())


def calculate_maximum(df: pd.DataFrame, column: str) -> float:
    """
    Find the maximum value in a numeric column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric.")

    return float(df[column].max())


def group_and_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> pd.DataFrame:
    """
    Group the dataset by one column and calculate
    the sum of another column.
    """

    if group_column not in df.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    result = (
        df.groupby(group_column)[value_column]
        .sum()
        .reset_index()
        .sort_values(
            by=value_column,
            ascending=False
        )
    )

    return result.reset_index(drop=True)