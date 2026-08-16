from tools.data_loader import load_csv
from tools.schema import get_dataset_schema


def test_dataset_schema():

    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    assert schema["row_count"] == 28
    assert schema["column_count"] == 5


def test_schema_contains_semantic_types():

    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    for column in schema["columns"]:
        assert "semantic_type" in column


def test_numeric_column_is_detected():

    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    revenue = next(
        column
        for column in schema["columns"]
        if column["name"] == "revenue"
    )

    assert revenue["semantic_type"] == "numeric"


def test_categorical_column_is_detected():

    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    region = next(
        column
        for column in schema["columns"]
        if column["name"] == "region"
    )

    assert region["semantic_type"] == "categorical"


def test_datetime_column_is_detected():

    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    date_column = next(
        column
        for column in schema["columns"]
        if column["name"] == "date"
    )

    assert date_column["semantic_type"] == "datetime"


def test_identifier_column_is_detected():

    df = load_csv(
        r"C:\Users\sanja\Downloads\titanic.csv"
    )

    schema = get_dataset_schema(df)

    passenger_id = next(
        column
        for column in schema["columns"]
        if column["name"] == "PassengerId"
    )

    assert passenger_id["semantic_type"] == "identifier"