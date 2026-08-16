from pathlib import Path

import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or is not a CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Only CSV files are currently supported."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The CSV file is empty."
        )

    return df