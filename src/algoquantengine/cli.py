import argparse
from pathlib import Path

from algoquantengine import graph
from algoquantengine.data.loaders import load_prices_csv
from algoquantengine.data.preprocess import clean_prices, compute_returns
from algoquantengine.data.features import estimate_mu_cov, corr_matrix

from algoquantengine.graph.build import build_graph_from_corr
from algoquantengine.graph.algorithms import compute_mst, pagerank_centrality, spectral_clusters_from_corr
from algoquantengine.report.plots import plot_mst, plot_cluster_heatmap
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)