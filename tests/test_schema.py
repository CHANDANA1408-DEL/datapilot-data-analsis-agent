from tools.data_loader import load_csv
from tools.schema import get_dataset_schema


def test_dataset_schema():
    df = load_csv("data/sales_data.csv")

    schema = get_dataset_schema(df)

    assert schema["row_count"] == 28
    assert schema["column_count"] == 5

    column_names = [
        column["name"]
        for column in schema["columns"]
    ]

    assert "date" in column_names
    assert "region" in column_names
    assert "product" in column_names
    assert "units_sold" in column_names
    assert "revenue" in column_names