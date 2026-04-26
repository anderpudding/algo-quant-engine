import pandas as pd
import pytest

from algoquantengine.data.validate import validate_price_frame


def test_validate_price_frame_accepts_valid_data():
    prices = pd.DataFrame(
        {
            "A": [100.0] * 30,
            "B": [101.0] * 30,
            "C": [102.0] * 30,
            "D": [103.0] * 30,
            "E": [104.0] * 30,
        }
    )

    validate_price_frame(prices, min_rows=30, min_assets=5)


def test_validate_price_frame_rejects_too_few_assets():
    prices = pd.DataFrame(
        {
            "A": [100.0] * 30,
            "B": [101.0] * 30,
        }
    )

    with pytest.raises(ValueError, match="assets"):
        validate_price_frame(prices, min_rows=30, min_assets=5)


def test_validate_price_frame_rejects_negative_prices():
    prices = pd.DataFrame(
        {
            "A": [100.0] * 30,
            "B": [101.0] * 30,
            "C": [102.0] * 30,
            "D": [103.0] * 30,
            "E": [-1.0] * 30,
        }
    )

    with pytest.raises(ValueError, match="positive"):
        validate_price_frame(prices, min_rows=30, min_assets=5)