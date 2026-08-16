from tools.data_loader import load_csv
from tools.data_analysis_engine import analyze_dataset


def test_total_revenue():

    df = load_csv("data/sales_data.csv")

    result = analyze_dataset(
        df,
        operation="aggregate",
        column="revenue",
        aggregation="sum",
    )

    assert result == 2834000.0


def test_average_revenue():

    df = load_csv("data/sales_data.csv")

    result = analyze_dataset(
        df,
        operation="aggregate",
        column="revenue",
        aggregation="mean",
    )

    assert round(result, 2) == 101214.29


def test_group_revenue_by_region():

    df = load_csv("data/sales_data.csv")

    result = analyze_dataset(
        df,
        operation="group",
        group_by="region",
        column="revenue",
        aggregation="sum",
    )

    assert result.iloc[0]["region"] == "North"
    assert result.iloc[0]["revenue"] == 765000


def test_filter_aggregate():

    df = load_csv("data/sales_data.csv")

    result = analyze_dataset(
        df,
        operation="filter_aggregate",
        filters={
            "product": "Laptop"
        },
        column="revenue",
        aggregation="sum",
    )

    assert result == 1340000.0


def test_rank():

    df = load_csv("data/sales_data.csv")

    result = analyze_dataset(
        df,
        operation="rank",
        column="revenue",
        top_n=3,
    )

    assert len(result) == 3
    assert result.iloc[0]["revenue"] == 180000