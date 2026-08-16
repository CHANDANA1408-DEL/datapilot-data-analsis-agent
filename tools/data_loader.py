import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV/TSV-style dataset with automatic delimiter detection.

    Supports common delimiters such as:
    - comma ,
    - tab \t
    - semicolon ;
    - pipe |
    """

    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
        )

    except Exception as error:
        raise ValueError(
            f"Could not load dataset '{path}': {error}"
        ) from error

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    return df