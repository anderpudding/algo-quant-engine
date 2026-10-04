import pandas as pd
import matplotlib.pyplot as plt

from algoquantengine.report.compare_plots import export_strategy_dashboard


def test_export_strategy_dashboard(tmp_path, monkeypatch):
    hybrid = "Hybrid Graph-Constrained"
    df = pd.DataFrame(
        {
            "strategy": ["Equal Weight", "Mean-Variance", "Min Variance", hybrid],
            "return": [0.10, 0.12, 0.08, 0.11],
            "volatility": [0.15, 0.18, 0.12, 0.16],
            "max_drawdown": [0.08, 0.10, 0.06, 0.09],
            "net_return": [0.095, 0.11, 0.075, 0.10],
        }
    )

    savefig = plt.savefig
    plotted_labels = []

    def capture_labels(*args, **kwargs):
        ax = plt.gca()
        plotted_labels.append([text.get_text() for text in ax.get_xticklabels() + list(ax.texts)])
        savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", capture_labels)
    export_strategy_dashboard(df, str(tmp_path))

    assert len(plotted_labels) == 4
    assert all(hybrid in labels for labels in plotted_labels)
    assert (tmp_path / "strategy_return_bar.png").exists()
    assert (tmp_path / "strategy_risk_return.png").exists()
    assert (tmp_path / "strategy_drawdown_bar.png").exists()
    assert (tmp_path / "strategy_net_vs_gross.png").exists()
