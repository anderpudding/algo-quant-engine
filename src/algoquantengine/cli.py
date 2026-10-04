from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algoquantengine.data.loaders import load_prices_csv
from algoquantengine.data.preprocess import clean_prices, compute_returns
from algoquantengine.data.features import estimate_mu_cov, corr_matrix
from algoquantengine.data.validate import validate_price_frame

from algoquantengine.graph.build import build_graph_from_corr
from algoquantengine.graph.algorithms import (
    compute_mst,
    pagerank_centrality,
    spectral_clusters_from_corr,
)
from algoquantengine.graph.constraints import cluster_weight_caps

from algoquantengine.opt.mean_variance import efficient_frontier
from algoquantengine.opt.risk import portfolio_pnl_from_scenarios, var_cvar, max_drawdown
from algoquantengine.opt.backtest import backtest_rebalance
from algoquantengine.opt.strategies import (
    equal_weight,
    mean_variance_best_sharpe,
    min_variance,
    hybrid_graph_constrained,
)

from algoquantengine.sim.scenarios import bootstrap_return_scenarios

from algoquantengine.report.plots import (
    plot_frontier,
    plot_mst,
    plot_cluster_heatmap,
)
from algoquantengine.report.export import (
    export_frontier_csv,
    export_weights_csv,
    export_group_caps_json,
    export_report_json,
)
from algoquantengine.report.compare import (
    evaluate_strategy,
    build_comparison_table,
    static_backtest,
)
from algoquantengine.report.compare_plots import export_strategy_dashboard

from algoquantengine.bench.scaling import benchmark_scaling
from algoquantengine.bench.plot import plot_scaling

from algoquantengine.report.rolling_compare import run_rolling_strategy_comparison
from algoquantengine.report.rolling_plots import (
    plot_rolling_equity_curves,
    plot_rolling_drawdowns,
)


def load_clean_validate_prices(args: argparse.Namespace) -> pd.DataFrame:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    if getattr(args, "validate_data", False):
        validate_price_frame(
            prices,
            min_rows=args.min_rows,
            min_assets=args.min_assets,
        )

    return prices


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--assets", type=int, default=None)
    parser.add_argument("--returns", choices=["log", "simple"], default="log")
    parser.add_argument("--annualize", type=int, default=252)
    parser.add_argument("--drop-thresh", type=float, default=0.05)

    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-assets", type=int, default=5)


def cmd_run(args: argparse.Namespace) -> None:
    prices = load_clean_validate_prices(args)

    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)
    corr = corr_matrix(rets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mu.txt").write_text("\n".join(map(str, mu.tolist())))
    (out_dir / "cov_shape.txt").write_text(f"{cov.shape}\n")
    (out_dir / "corr_shape.txt").write_text(f"{corr.shape}\n")

    print("OK")
    print(f"Assets: {rets.shape[1]}, Observations: {rets.shape[0]}")
    print(f"Outputs written to: {out_dir}")


def cmd_graph(args: argparse.Namespace) -> None:
    prices = load_clean_validate_prices(args)

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    corr = corr_matrix(rets)

    G = build_graph_from_corr(tickers, corr, threshold=args.threshold)
    mst = compute_mst(G, weight="dist")
    pr = pagerank_centrality(G, weight="rho")

    n = len(tickers)
    k = min(max(args.clusters, 2), n)
    labels = spectral_clusters_from_corr(corr, n_clusters=k, seed=42)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    plot_mst(mst, str(fig_dir / "mst.png"))
    plot_cluster_heatmap(corr, labels, str(fig_dir / "cluster_heatmap.png"))

    pd.DataFrame({"ticker": tickers, "cluster": labels}).to_csv(
        tab_dir / "clusters.csv", index=False
    )
    pd.DataFrame({"ticker": list(pr.keys()), "pagerank": list(pr.values())}).to_csv(
        tab_dir / "pagerank.csv", index=False
    )

    print("OK")
    print(f"Saved: {fig_dir / 'mst.png'}, {fig_dir / 'cluster_heatmap.png'}")
    print(f"Saved: {tab_dir / 'clusters.csv'}, {tab_dir / 'pagerank.csv'}")


def cmd_opt(args: argparse.Namespace) -> None:
    prices = load_clean_validate_prices(args)

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)

    frontier = efficient_frontier(cov, mu, n_points=args.frontier)
    best = max(frontier, key=lambda p: p["sharpe"])
    w_best = best["weights"]

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    export_frontier_csv(frontier, str(tab_dir / "frontier.csv"))
    export_weights_csv(tickers, w_best, str(tab_dir / "weights_best_sharpe.csv"))
    plot_frontier(frontier, str(fig_dir / "frontier.png"))

    print("OK")
    print(f"Saved: {tab_dir / 'frontier.csv'}")
    print(f"Saved: {tab_dir / 'weights_best_sharpe.csv'}")
    print(f"Saved: {fig_dir / 'frontier.png'}")


def cmd_hybrid(args: argparse.Namespace) -> None:
    prices = load_clean_validate_prices(args)

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)

    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)
    corr = corr_matrix(rets)

    n = len(tickers)
    k = min(max(args.clusters, 2), n)
    labels = spectral_clusters_from_corr(corr, n_clusters=k, seed=42)

    caps = cluster_weight_caps(labels, max_per_cluster=args.cap)

    frontier = efficient_frontier(
        Sigma=cov,
        mu=mu,
        n_points=args.frontier,
        extra_caps=caps,
    )

    best = max(frontier, key=lambda p: p["sharpe"])
    w_best = best["weights"]

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"ticker": tickers, "cluster": labels}).to_csv(
        tab_dir / "clusters.csv", index=False
    )
    export_group_caps_json(caps, str(tab_dir / "cluster_caps.json"))

    export_frontier_csv(frontier, str(tab_dir / "frontier_capped.csv"))
    export_weights_csv(tickers, w_best, str(tab_dir / "weights_best_sharpe_capped.csv"))
    plot_frontier(frontier, str(fig_dir / "frontier_capped.png"))

    print("OK")
    print(f"Saved: {tab_dir / 'clusters.csv'}")
    print(f"Saved: {tab_dir / 'cluster_caps.json'}")
    print(f"Saved: {tab_dir / 'frontier_capped.csv'}")
    print(f"Saved: {tab_dir / 'weights_best_sharpe_capped.csv'}")
    print(f"Saved: {fig_dir / 'frontier_capped.png'}")


def cmd_risk(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_clean_validate_prices(args)

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)

    frontier = efficient_frontier(cov, mu, n_points=args.frontier)
    best = max(frontier, key=lambda p: p["sharpe"])
    w = best["weights"]

    scen = bootstrap_return_scenarios(
        rets,
        horizon=args.horizon,
        n_paths=args.paths,
        seed=42,
    )
    pnl = portfolio_pnl_from_scenarios(scen, w)
    var95, cvar95 = var_cvar(pnl, alpha=args.alpha)

    def make_w(window_returns: pd.DataFrame):
        mu_w, cov_w = estimate_mu_cov(window_returns, annualize=args.annualize)
        fr = efficient_frontier(cov_w, mu_w, n_points=max(10, args.frontier // 2))
        b = max(fr, key=lambda p: p["sharpe"])
        return b["weights"]

    equity_result = backtest_rebalance(
        prices=prices,
        rebalance_every=args.rebalance,
        lookback=args.lookback,
        make_weights_fn=make_w,
    )

    if isinstance(equity_result, tuple):
        equity = equity_result[0]
    else:
        equity = equity_result

    mdd = max_drawdown(equity.to_numpy())

    report = {
        "n_assets": len(tickers),
        "n_obs_returns": int(rets.shape[0]),
        "params": {
            "alpha": args.alpha,
            "paths": args.paths,
            "horizon": args.horizon,
            "rebalance": args.rebalance,
            "lookback": args.lookback,
        },
        "best_portfolio": {
            "return": float(best["return"]),
            "vol": float(best["vol"]),
            "sharpe": float(best["sharpe"]),
        },
        "risk": {
            "VaR": float(var95),
            "CVaR": float(cvar95),
            "max_drawdown": float(mdd),
        },
    }

    export_report_json(report, str(out_dir / "risk_report.json"))
    print("OK")
    print(f"Saved: {out_dir / 'risk_report.json'}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    sizes = [10, 20, 40, 80, 120]
    df = benchmark_scaling(sizes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scaling.csv"
    fig_path = out_dir / "scaling.png"

    df.to_csv(csv_path, index=False)
    plot_scaling(df, fig_path)

    print("OK")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


def cmd_demo(args: argparse.Namespace) -> None:
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    shared = {
        "data": args.data,
        "date_col": args.date_col,
        "assets": args.assets,
        "returns": args.returns,
        "annualize": args.annualize,
        "drop_thresh": args.drop_thresh,
        "validate_data": args.validate_data,
        "min_rows": args.min_rows,
        "min_assets": args.min_assets,
    }

    cmd_graph(
        argparse.Namespace(
            **shared,
            threshold=args.threshold,
            clusters=args.clusters,
            out_dir=str(base_out / "graph"),
        )
    )

    cmd_hybrid(
        argparse.Namespace(
            **shared,
            clusters=args.clusters,
            cap=args.cap,
            frontier=args.frontier,
            out_dir=str(base_out / "hybrid"),
        )
    )

    cmd_risk(
        argparse.Namespace(
            **shared,
            frontier=max(10, args.frontier),
            alpha=args.alpha,
            paths=args.paths,
            horizon=args.horizon,
            rebalance=args.rebalance,
            lookback=args.lookback,
            out_dir=str(base_out / "risk"),
        )
    )

    if args.benchmark:
        cmd_benchmark(argparse.Namespace(out_dir=str(base_out / "benchmarks")))

    print("OK")
    print(f"Saved demo outputs to: {base_out}")


def cmd_compare(args: argparse.Namespace) -> None:
    prices = load_clean_validate_prices(args)

    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)

    n = len(prices.columns)

    w_eq = equal_weight(n)
    w_mv = mean_variance_best_sharpe(cov, mu)
    w_min = min_variance(cov)
    w_hybrid, _, _ = hybrid_graph_constrained(
        cov,
        mu,
        corr_matrix(rets),
        n_clusters=args.clusters,
        max_per_cluster=args.cap,
        seed=args.seed,
    )

    strategies = [
        ("Equal Weight", w_eq),
        ("Mean-Variance", w_mv),
        ("Min Variance", w_min),
        ("Hybrid Graph-Constrained", w_hybrid),
    ]

    results = []

    for name, w in strategies:
        res = evaluate_strategy(
            name=name,
            weights=w,
            rets=rets,
            prices=prices,
            paths=args.paths,
            horizon=args.horizon,
            alpha=args.alpha,
            backtest_fn=lambda p, w=w: static_backtest(p, w),
            cost_rate=args.cost_rate,
        )
        results.append(res)

    df = build_comparison_table(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "strategy_comparison.csv", index=False)

    fig_dir = out_dir / "figures"
    export_strategy_dashboard(df, str(fig_dir))

    print("OK")
    print(f"Saved: {out_dir / 'strategy_comparison.csv'}")
    print(f"Saved figures to: {fig_dir}")
    print(df)

def cmd_rolling_compare(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    if getattr(args, "validate_data", False):
        validate_price_frame(prices, min_rows=args.min_rows, min_assets=args.min_assets)

    metrics_df, equity_df = run_rolling_strategy_comparison(
        prices=prices,
        lookback=args.lookback,
        rebalance=args.rebalance,
        transaction_cost=args.cost,
    )

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / "rolling_strategy_metrics.csv", index=False)
    equity_df.to_csv(out_dir / "rolling_equity_curves.csv")

    plot_rolling_equity_curves(
        equity_df,
        str(fig_dir / "rolling_equity_curves.png"),
    )
    plot_rolling_drawdowns(
        equity_df,
        str(fig_dir / "rolling_drawdowns.png"),
    )

    print("OK")
    print(f"Saved: {out_dir / 'rolling_strategy_metrics.csv'}")
    print(f"Saved: {out_dir / 'rolling_equity_curves.csv'}")
    print(f"Saved figures to: {fig_dir}")
    print(metrics_df)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="algoquantengine")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run data pipeline: prices -> returns -> mu/cov/corr")
    add_data_args(run)
    run.add_argument("--out-dir", default="outputs/reports/run0")
    run.set_defaults(func=cmd_run)

    graph = sub.add_parser("graph", help="Build correlation graph, MST, centrality, and clusters")
    add_data_args(graph)
    graph.add_argument("--threshold", type=float, default=0.3, help="Keep edges with |corr| >= threshold")
    graph.add_argument("--clusters", type=int, default=6)
    graph.add_argument("--out-dir", default="outputs/reports/graph0")
    graph.set_defaults(func=cmd_graph)

    opt = sub.add_parser("opt", help="Run mean-variance optimization and efficient frontier")
    add_data_args(opt)
    opt.add_argument("--frontier", type=int, default=25)
    opt.add_argument("--out-dir", default="outputs/reports/opt0")
    opt.set_defaults(func=cmd_opt)

    hy = sub.add_parser("hybrid", help="Hybrid run: graph clusters -> caps -> constrained frontier")
    add_data_args(hy)
    hy.add_argument("--clusters", type=int, default=6)
    hy.add_argument("--cap", type=float, default=0.25, help="Max total weight per cluster")
    hy.add_argument("--frontier", type=int, default=25)
    hy.add_argument("--out-dir", default="outputs/reports/hybrid0")
    hy.set_defaults(func=cmd_hybrid)

    risk = sub.add_parser("risk", help="Compute VaR/CVaR and drawdown from backtest")
    add_data_args(risk)
    risk.add_argument("--frontier", type=int, default=25)
    risk.add_argument("--alpha", type=float, default=0.95)
    risk.add_argument("--paths", type=int, default=2000)
    risk.add_argument("--horizon", type=int, default=10)
    risk.add_argument("--rebalance", type=int, default=21)
    risk.add_argument("--lookback", type=int, default=252)
    risk.add_argument("--out-dir", default="outputs/reports/risk0")
    risk.set_defaults(func=cmd_risk)

    bench = sub.add_parser("benchmark", help="Run runtime scaling benchmarks")
    bench.add_argument("--out-dir", default="outputs/benchmarks")
    bench.set_defaults(func=cmd_benchmark)

    demo = sub.add_parser("demo", help="Run full end-to-end demo pipeline")
    add_data_args(demo)
    demo.add_argument("--threshold", type=float, default=0.3)
    demo.add_argument("--clusters", type=int, default=6)
    demo.add_argument("--cap", type=float, default=0.25)
    demo.add_argument("--frontier", type=int, default=25)
    demo.add_argument("--alpha", type=float, default=0.95)
    demo.add_argument("--paths", type=int, default=2000)
    demo.add_argument("--horizon", type=int, default=10)
    demo.add_argument("--rebalance", type=int, default=21)
    demo.add_argument("--lookback", type=int, default=252)
    demo.add_argument("--benchmark", action="store_true")
    demo.add_argument("--out-dir", default="outputs/demo")
    demo.set_defaults(func=cmd_demo)

    cmp = sub.add_parser("compare", help="Compare portfolio strategies")
    add_data_args(cmp)
    cmp.add_argument("--paths", type=int, default=1000)
    cmp.add_argument("--horizon", type=int, default=10)
    cmp.add_argument("--alpha", type=float, default=0.95)
    cmp.add_argument("--out-dir", default="outputs/reports/compare")
    cmp.add_argument("--cost-rate", type=float, default=0.001)
    cmp.add_argument("--clusters", type=int, default=4)
    cmp.add_argument("--cap", type=float, default=0.40, help="Max total weight per cluster")
    cmp.add_argument("--seed", type=int, default=42)
    cmp.set_defaults(func=cmd_compare)

    roll = sub.add_parser("rolling-compare", help="Run rolling-window strategy comparison")
    roll.add_argument("--data", required=True)
    roll.add_argument("--date-col", default="Date")
    roll.add_argument("--assets", type=int, default=None)
    roll.add_argument("--drop-thresh", type=float, default=0.05)
    roll.add_argument("--lookback", type=int, default=60)
    roll.add_argument("--rebalance", type=int, default=21)
    roll.add_argument("--cost", type=float, default=0.001, help="Transaction cost per unit turnover")
    roll.add_argument("--validate-data", action="store_true")
    roll.add_argument("--min-rows", type=int, default=90)
    roll.add_argument("--min-assets", type=int, default=5)
    roll.add_argument("--out-dir", default="outputs/reports/rolling_compare")
    roll.set_defaults(func=cmd_rolling_compare)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
