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
def filter_data(
    df: pd.DataFrame,
    column: str,
    value
) -> pd.DataFrame:
    """
    Filter rows where a column equals a specified value.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    result = df[df[column] == value].copy()

    return result.reset_index(drop=True)
def rank_by_column(
    df: pd.DataFrame,
    column: str,
    ascending: bool = False,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Rank rows based on a numeric column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    return (
        df.sort_values(
            by=column,
            ascending=ascending
        )
        .head(top_n)
        .reset_index(drop=True)
    )
def convert_to_datetime(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Convert a column to datetime.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    result = df.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce"
    )

    if result[column].isna().any():
        raise ValueError(
            f"Column '{column}' contains invalid dates."
        )

    return result
def add_quarter_column(
    df: pd.DataFrame,
    date_column: str
) -> pd.DataFrame:
    """
    Add a quarter column based on a date column.
    """

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' does not exist."
        )

    result = convert_to_datetime(
        df,
        date_column
    )

    result["quarter"] = (
        result[date_column]
        .dt.to_period("Q")
        .astype(str)
    )

    return result
def quarterly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str
) -> pd.DataFrame:
    """
    Calculate the sum of a numeric column by quarter.
    """

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[value_column]
    ):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    result = add_quarter_column(
        df,
        date_column
    )

    quarterly = (
        result.groupby("quarter")[value_column]
        .sum()
        .reset_index()
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    return quarterly
def calculate_growth(
    df: pd.DataFrame,
    period_column: str,
    value_column: str
) -> pd.DataFrame:
    """
    Calculate period-over-period percentage growth.
    """

    if period_column not in df.columns:
        raise ValueError(
            f"Column '{period_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    result = df.sort_values(
        by=period_column
    ).copy()

    result["growth_percentage"] = (
        result[value_column]
        .pct_change()
        .mul(100)
    )

    return result.reset_index(drop=True)