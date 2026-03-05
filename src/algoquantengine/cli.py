import argparse
from pathlib import Path

from algoquantengine import graph
from algoquantengine.data.loaders import load_prices_csv
from algoquantengine.data.preprocess import clean_prices, compute_returns
from algoquantengine.data.features import estimate_mu_cov, corr_matrix

from algoquantengine.opt import risk
from algoquantengine.opt.mean_variance import efficient_frontier
from algoquantengine.report.plots import plot_frontier
from algoquantengine.report.export import export_frontier_csv, export_weights_csv

from algoquantengine.graph.build import build_graph_from_corr
from algoquantengine.graph.algorithms import compute_mst, pagerank_centrality, spectral_clusters_from_corr
from algoquantengine.report.plots import plot_mst, plot_cluster_heatmap

from algoquantengine.graph.algorithms import spectral_clusters_from_corr
from algoquantengine.graph.constraints import cluster_weight_caps
from algoquantengine.report.export import export_group_caps_json

from algoquantengine.sim.scenarios import bootstrap_return_scenarios
from algoquantengine.opt.risk import portfolio_pnl_from_scenarios, var_cvar, max_drawdown
from algoquantengine.opt.backtest import backtest_rebalance
from algoquantengine.report.export import export_report_json

from algoquantengine.bench.scaling import benchmark_scaling
from algoquantengine.bench.plot import plot_scaling

import pandas as pd
import numpy as np


def cmd_run(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)
    corr = corr_matrix(rets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save minimal artifacts for now
    (out_dir / "mu.txt").write_text("\n".join(map(str, mu.tolist())))
    (out_dir / "cov_shape.txt").write_text(f"{cov.shape}\n")
    (out_dir / "corr_shape.txt").write_text(f"{corr.shape}\n")

    print("OK")
    print(f"Assets: {rets.shape[1]}, Observations: {rets.shape[0]}")
    print(f"Outputs written to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="algoquantengine")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run data pipeline: prices -> returns -> mu/cov/corr")
    run.add_argument("--data", required=True, help="Path to CSV containing prices")
    run.add_argument("--date-col", default="Date", help="Name of date column")
    run.add_argument("--assets", type=int, default=None, help="Use first N asset columns")
    run.add_argument("--returns", choices=["log", "simple"], default="log")
    run.add_argument("--annualize", type=int, default=252)
    run.add_argument("--drop-thresh", type=float, default=0.05)
    run.add_argument("--out-dir", default="outputs/reports/run0")
    run.set_defaults(func=cmd_run)

    graph = sub.add_parser("graph", help="Build correlation graph, MST, centrality, and clusters")
    graph.add_argument("--data", required=True)
    graph.add_argument("--date-col", default="Date")
    graph.add_argument("--assets", type=int, default=None)
    graph.add_argument("--returns", choices=["log", "simple"], default="log")
    graph.add_argument("--annualize", type=int, default=252)
    graph.add_argument("--drop-thresh", type=float, default=0.05)
    graph.add_argument("--threshold", type=float, default=0.3, help="Keep edges with |corr| >= threshold")
    graph.add_argument("--clusters", type=int, default=6)
    graph.add_argument("--out-dir", default="outputs/reports/graph0")
    graph.set_defaults(func=cmd_graph)

    opt = sub.add_parser("opt", help="Run mean-variance optimization and efficient frontier")
    opt.add_argument("--data", required=True)
    opt.add_argument("--date-col", default="Date")
    opt.add_argument("--assets", type=int, default=None)
    opt.add_argument("--returns", choices=["log", "simple"], default="log")
    opt.add_argument("--annualize", type=int, default=252)
    opt.add_argument("--drop-thresh", type=float, default=0.05)
    opt.add_argument("--frontier", type=int, default=25)
    opt.add_argument("--out-dir", default="outputs/reports/opt0")
    opt.set_defaults(func=cmd_opt)

    hy = sub.add_parser("hybrid", help="Hybrid run: graph clusters -> caps -> constrained frontier")
    hy.add_argument("--data", required=True)
    hy.add_argument("--date-col", default="Date")
    hy.add_argument("--assets", type=int, default=None)
    hy.add_argument("--returns", choices=["log", "simple"], default="log")
    hy.add_argument("--annualize", type=int, default=252)
    hy.add_argument("--drop-thresh", type=float, default=0.05)
    hy.add_argument("--clusters", type=int, default=6)
    hy.add_argument("--cap", type=float, default=0.25, help="Max total weight per cluster")
    hy.add_argument("--frontier", type=int, default=25)
    hy.add_argument("--out-dir", default="outputs/reports/hybrid0")
    hy.set_defaults(func=cmd_hybrid)

    risk = sub.add_parser("risk", help="Compute VaR/CVaR and drawdown from backtest")
    risk.add_argument("--data", required=True)
    risk.add_argument("--date-col", default="Date")
    risk.add_argument("--assets", type=int, default=None)
    risk.add_argument("--returns", choices=["log", "simple"], default="log")
    risk.add_argument("--annualize", type=int, default=252)
    risk.add_argument("--drop-thresh", type=float, default=0.05)
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

    return p

def cmd_graph(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    corr = corr_matrix(rets)

    G = build_graph_from_corr(tickers, corr, threshold=args.threshold)
    mst = compute_mst(G, weight="dist")
    pr = pagerank_centrality(G, weight="rho")
    labels = spectral_clusters_from_corr(corr, n_clusters=args.clusters, seed=42)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    plot_mst(mst, str(fig_dir / "mst.png"))
    plot_cluster_heatmap(corr, labels, str(fig_dir / "cluster_heatmap.png"))

    pd.DataFrame({"ticker": tickers, "cluster": labels}).to_csv(tab_dir / "clusters.csv", index=False)
    pd.DataFrame({"ticker": list(pr.keys()), "pagerank": list(pr.values())}).to_csv(tab_dir / "pagerank.csv", index=False)

    print("OK")
    print(f"Saved: {fig_dir/'mst.png'}, {fig_dir/'cluster_heatmap.png'}")
    print(f"Saved: {tab_dir/'clusters.csv'}, {tab_dir/'pagerank.csv'}")

def cmd_opt(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)

    frontier = efficient_frontier(cov, mu, n_points=args.frontier)

    # pick best Sharpe
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
    print(f"Saved: {tab_dir/'frontier.csv'}")
    print(f"Saved: {tab_dir/'weights_best_sharpe.csv'}")
    print(f"Saved: {fig_dir/'frontier.png'}")

def cmd_hybrid(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)

    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)
    corr = corr_matrix(rets)

    n = len(tickers)
    # robust k selection
    k = min(max(args.clusters, 2), n)
    labels = spectral_clusters_from_corr(corr, n_clusters=k, seed=42)

    caps = cluster_weight_caps(labels, max_per_cluster=args.cap)

    # run frontier with caps
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

    # exports
    import pandas as pd
    pd.DataFrame({"ticker": tickers, "cluster": labels}).to_csv(tab_dir / "clusters.csv", index=False)
    export_group_caps_json(caps, str(tab_dir / "cluster_caps.json"))

    export_frontier_csv(frontier, str(tab_dir / "frontier_capped.csv"))
    export_weights_csv(tickers, w_best, str(tab_dir / "weights_best_sharpe_capped.csv"))
    plot_frontier(frontier, str(fig_dir / "frontier_capped.png"))

    print("OK")
    print(f"Saved: {tab_dir/'clusters.csv'}")
    print(f"Saved: {tab_dir/'cluster_caps.json'}")
    print(f"Saved: {tab_dir/'frontier_capped.csv'}")
    print(f"Saved: {tab_dir/'weights_best_sharpe_capped.csv'}")
    print(f"Saved: {fig_dir/'frontier_capped.png'}")

def cmd_risk(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.data, date_col=args.date_col)
    prices = clean_prices(prices, fill_method="ffill", drop_thresh=args.drop_thresh)

    if args.assets is not None:
        prices = prices.iloc[:, : args.assets]

    tickers = list(prices.columns)
    rets = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(rets, annualize=args.annualize)

    # baseline weights: best sharpe from unconstrained frontier
    frontier = efficient_frontier(cov, mu, n_points=args.frontier)
    best = max(frontier, key=lambda p: p["sharpe"])
    w = best["weights"]

    # VaR/CVaR via bootstrap scenarios
    scen = bootstrap_return_scenarios(rets, horizon=args.horizon, n_paths=args.paths, seed=42)
    pnl = portfolio_pnl_from_scenarios(scen, w)
    var95, cvar95 = var_cvar(pnl, alpha=args.alpha)

    # drawdown via backtest (recompute weights at each rebalance from window)
    def make_w(window_returns):
        mu_w, cov_w = estimate_mu_cov(window_returns, annualize=args.annualize)
        fr = efficient_frontier(cov_w, mu_w, n_points=max(10, args.frontier // 2))
        b = max(fr, key=lambda p: p["sharpe"])
        return b["weights"]

    equity = backtest_rebalance(
        prices=prices,
        rebalance_every=args.rebalance,
        lookback=args.lookback,
        make_weights_fn=make_w,
    )
    mdd = max_drawdown(equity.to_numpy())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Saved: {out_dir/'risk_report.json'}")

def cmd_benchmark(args):

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

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)