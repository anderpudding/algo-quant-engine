import pandas as pd

from algoquantengine.report.compare_plots import export_strategy_dashboard


def test_export_strategy_dashboard(tmp_path):
    df = pd.DataFrame(
        {
            "strategy": ["Equal Weight", "Mean-Variance"],
            "return": [0.10, 0.12],
            "volatility": [0.15, 0.18],
            "max_drawdown": [0.08, 0.10],
            "net_return": [0.095, 0.11],
        }
    )

    export_strategy_dashboard(df, str(tmp_path))

    assert (tmp_path / "strategy_return_bar.png").exists()
    assert (tmp_path / "strategy_risk_return.png").exists()
    assert (tmp_path / "strategy_drawdown_bar.png").exists()
    assert (tmp_path / "strategy_net_vs_gross.png").exists()