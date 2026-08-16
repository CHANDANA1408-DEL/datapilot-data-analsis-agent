import pandas as pd
import pytest

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


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-04-01",
            "2026-07-01",
            "2026-10-01",
        ],
        "region": [
            "North",
            "South",
            "North",
            "South",
        ],
        "product": [
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
        ],
        "units_sold": [
            10,
            20,
            30,
            40,
        ],
        "revenue": [
            100000,
            200000,
            300000,
            400000,
        ],
    })


def test_calculate_total(sample_dataframe):
    assert calculate_total(
        sample_dataframe,
        "revenue"
    ) == 1000000


def test_calculate_average(sample_dataframe):
    assert calculate_average(
        sample_dataframe,
        "revenue"
    ) == 250000


def test_calculate_minimum(sample_dataframe):
    assert calculate_minimum(
        sample_dataframe,
        "revenue"
    ) == 100000


def test_calculate_maximum(sample_dataframe):
    assert calculate_maximum(
        sample_dataframe,
        "revenue"
    ) == 400000


def test_group_and_sum(sample_dataframe):
    result = group_and_sum(
        sample_dataframe,
        "region",
        "revenue"
    )

    assert result.iloc[0]["region"] == "South"
    assert result.iloc[0]["revenue"] == 600000


def test_filter_data(sample_dataframe):
    result = filter_data(
        sample_dataframe,
        "region",
        "North"
    )

    assert len(result) == 2
    assert all(result["region"] == "North")


def test_rank_by_column(sample_dataframe):
    result = rank_by_column(
        sample_dataframe,
        "revenue",
        top_n=2
    )

    assert len(result) == 2
    assert result.iloc[0]["revenue"] == 400000
    assert result.iloc[1]["revenue"] == 300000


def test_quarterly_sum(sample_dataframe):
    result = quarterly_sum(
        sample_dataframe,
        "date",
        "revenue"
    )

    assert len(result) == 4
    assert result.iloc[0]["quarter"] == "2026Q1"
    assert result.iloc[0]["revenue"] == 100000


def test_calculate_growth(sample_dataframe):
    quarterly = quarterly_sum(
        sample_dataframe,
        "date",
        "revenue"
    )

    result = calculate_growth(
        quarterly,
        "quarter",
        "revenue"
    )

    assert pd.isna(
        result.iloc[0]["growth_percentage"]
    )

    assert result.iloc[1]["growth_percentage"] == pytest.approx(
        100.0
    )

    assert result.iloc[2]["growth_percentage"] == pytest.approx(
        50.0
    )