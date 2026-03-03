import argparse
from pathlib import Path

from algoquantengine.data.loaders import load_prices_csv
from algoquantengine.data.preprocess import clean_prices, compute_returns
from algoquantengine.data.features import estimate_mu_cov, corr_matrix


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

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)