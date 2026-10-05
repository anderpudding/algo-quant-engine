# AlgoQuantEngine

Hybrid quantitative finance + graph analytics system implementing portfolio optimization, correlation network analysis, risk metrics, and algorithmic benchmarking.

---

## Overview

AlgoQuantEngine is a computational research project that combines **quantitative finance**, **graph algorithms**, and **algorithmic performance analysis**.

The system integrates:

- Portfolio optimization (mean–variance framework)
- Asset correlation network analysis (MST, spectral clustering)
- Graph-based diversification constraints (cluster caps)
- Scenario-based risk estimation (VaR/CVaR)
- Walk-forward backtesting (drawdown)
- Runtime scaling benchmarks (complexity in practice)

The goal is to explore how **market network structure** can shape portfolio construction and how core components scale with **number of assets (N)**.

---

## System Architecture

```
           +---------------------+
           |   Market Data CSV   |
           +----------+----------+
                      |
                      v
           +----------------------+
           |     Data Pipeline    |
           |  (prices -> returns) |
           +----------+-----------+
                      |
        +-------------+-------------+
        |                           |
        v                           v
+---------------+           +----------------+
| Graph Engine  |           | Optimization   |
| (MST, network)|           | (Mean-Variance)|
+-------+-------+           +--------+-------+
        |                           |
        +-------------+-------------+
                      |
                      v
              +---------------+
              | Risk Analysis |
              | VaR / CVaR    |
              +-------+-------+
                      |
                      v
              +---------------+
              | Reports &     |
              | Visualizations|
              +---------------+
```

---

# Quickstart (2 minute demo)

Install the package in editable mode:

```bash
python -m pip install -e .
pip install -r requirements.txt
```

Run the full system pipeline:

```bash
python -m algoquantengine demo \
--data data/raw/prices_demo.csv \
--clusters 2 \
--cap 0.9 \
--lookback 3 \
--rebalance 1 \
--horizon 3 \
--paths 500 \
--benchmark
```

Outputs will be generated in:
```
output/demo/
```

Example output structure:

```
outputs/demo
├─ graph/
│  ├─ figures/
│  └─ tables/
├─ hybrid/
│  ├─ figures/
│  └─ tables/
├─ risk/
│  └─ risk_report.json
└─ benchmarks/
   ├─ scaling.csv
   └─ scaling.png
```

---

## Key Components

### Data Pipeline

Processes financial time series data and computes statistical features.

* Price ingestion
* Return calculation
* Covariance matrix
* Correlation matrix

### Portfolio Optimization

Implements classical portfolio theory.

Capabilities:

* Mean–variance optimization
* Efficient frontier generation
* Projected gradient descent optimizer
* Portfolio Sharpe ratio maximization

### Graph Analytics

Constructs correlation-based asset networks.

Implemented algorithms:

* Correlation network construction
* Minimum spanning tree (MST)
* Spectral clustering
* Graph-based diversification constraints

Graph structure is used to build **cluster weight caps** that constrain portfolio allocation.

### Risk Analysis

Implements standard quantitative risk metrics.

Metrics:

* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* Maximum drawdown
* Scenario-based Monte Carlo simulation
* Walk-forward portfolio backtesting

### Simulation

Generates potential market scenarios.

Methods include:

* Bootstrap sampling of historical returns
* Monte Carlo simulation of portfolio returns
* Scenario-based stress testing

### Benchmarking

Evaluates algorithmic scalability and runtime complexity.

Run benchmark suite:

```bash
python -m algoquantengine benchmark
```

Outputs:

```
outputs/benchmarks/
├─ scaling.csv
└─ scaling.png
```

The scaling experiment measures runtime of:

* Covariance computation
* Graph construction (MST)
* Portfolio optimization

---

## Algorithms Implemented

* Mean–Variance Portfolio Optimization
* Projected Gradient Descent
* Minimum Spanning Tree (Kruskal)
* Correlation Network Construction
* Spectral Clustering
* Bootstrap Monte Carlo Simulation

---

## Computational Complexity

| Component              | Complexity |
| ---------------------- | ---------- |
| Covariance computation | O(TN²)     |
| Gradient descent step  | O(N²)      |
| Simplex projection     | O(N log N) |
| Minimum spanning tree  | O(E log E) |

Where:

* **N** = number of assets
* **T** = number of time observations
* **E** = number of graph edges

Benchmark results illustrate empirical scaling of these components.

---

## Command Line Interface

Primary commands:

#### Graph analysis

```bash
python -m algoquantengine graph --data data/raw/prices_demo.csv
```

Generates correlation network and MST.

---

#### Hybrid portfolio optimization

```bash
python -m algoquantengine hybrid --data data/raw/prices_demo.csv
```

Constructs cluster constraints from graph structure and runs constrained optimization.

---

#### Risk analysis

```bash
python -m algoquantengine risk --data data/raw/prices_demo.csv
```

Computes:
* Var
* CVar
* Maximum drawdown

---

#### Benchmark scaling

```bash
python -m algoquantengine benchmark
```

Produces runtime scaling results.

---

#### Full pipeline demo

```bash
python -m algoquantengine demo --data data/raw/prices_demo.csv
```

Runs the complete system.

#### Strategy Visualization Dashboard

The strategy comparison command evaluates Equal Weight, Mean-Variance, Minimum
Variance (reported as `Min Variance`), and Hybrid Graph-Constrained portfolios
with the same metrics and exports these visual summaries:

- Strategy return bar chart
- Risk-return scatter plot
- Drawdown comparison
- Gross vs net return after transaction costs

Run:

```bash
python -m algoquantengine compare \
  --data data/raw/prices_demo.csv \
  --clusters 2 --cap 0.9 --seed 42 \
  --paths 500 \
  --horizon 5 \
  --out-dir outputs/reports/compare_demo
```

---

## Project Structure

```
algoquantengine
│
├─ README.md
├─ requirements.txt
│
├─ data/
│   ├─ raw
│   └─ processed
│
├─ outputs/
│   ├─ figures
│   └─ reports
│
├─ notebooks/
│
├─ src/
│   └─ algoquantengine/
│       ├─ data/
│       ├─ graph/
│       ├─ opt/
│       ├─ sim/
│       ├─ bench/
│       └─ report/
│
├─ tests/
│
└─ benchmarks/
```

---

## Reproducibility

Experiments are designed to be reproducible.

* Random seeds are centralized in configuration.
* Synthetic data generation uses deterministic seeds.
* Benchmark experiments use fixed parameters.

---

## Real Market Data Workflow

AlgoQuantEngine can run on larger historical price datasets with one date column and one column per asset.

Expected CSV format:

```csv
Date,AAPL,MSFT,NVDA,GOOG,AMZN
2021-01-04,129.41,217.69,13.08,86.41,159.33
```

Clean a raw CSV:

```
python scripts/prepare_prices_csv.py \
  --input data/raw/real/my_prices.csv \
  --output data/processed/my_prices_clean.csv
```

Run strategy comparison:

```
python -m algoquantengine compare \
  --data data/processed/my_prices_clean.csv \
  --validate-data \
  --min-rows 252 \
  --min-assets 20 \
  --paths 1000 \
  --horizon 10 \
  --out-dir outputs/reports/compare_real
```

---

## Rolling Strategy Comparison

Run walk-forward strategy comparison with periodic rebalancing and transaction costs:

The same four strategies are evaluated. At every Hybrid rebalance, correlation,
graph clusters, and cluster caps are recomputed from only the current historical
lookback window, excluding the traded date and future returns to avoid look-ahead
bias. Both comparison commands default to `--clusters 4 --cap 0.40 --seed 42`;
the effective cluster count times the cap must be at least 1. For the two-asset
demo, use `--clusters 2 --cap 0.9`.

```bash
python -m algoquantengine rolling-compare \
  --data data/raw/prices_demo.csv \
  --lookback 3 \
  --rebalance 1 \
  --clusters 2 --cap 0.9 --seed 42 \
  --cost 0.001 \
  --out-dir outputs/reports/rolling_compare_demo
```

Outputs:

```
rolling_strategy_metrics.csv
rolling_equity_curves.csv
figures/rolling_equity_curves.png
figures/rolling_drawdowns.png
```

Metrics include total return, annualized return, volatility, Sharpe ratio, max drawdown, average turnover, and transaction cost setting.

---

## Benchmark / Factor Exposure Analysis

Optionally evaluate every rolling strategy against an independent benchmark
price CSV: annualized regression alpha, beta, benchmark correlation, R², active
return, tracking error, information ratio, and residual volatility.

```bash
python -m algoquantengine rolling-compare \
  --data data/processed/my_prices_clean.csv \
  --benchmark-data data/processed/spy_prices.csv \
  --benchmark-column SPY \
  --lookback 126 \
  --rebalance 21 \
  --clusters 4 \
  --cap 0.35 \
  --cost 0.001 \
  --factor-window 60 \
  --out-dir outputs/reports/rolling_compare_real
```

The benchmark CSV uses the same date column (`Date` by default, configurable
with `--date-col`). Specify `--benchmark-column` when there are multiple numeric
columns; a single numeric price column is selected automatically. The benchmark
does not change the investable universe or portfolio weights. Without
`--benchmark-data`, existing outputs and behavior are unchanged.

Analysis uses realized net walk-forward equity returns after transaction costs,
not future information for portfolio construction. Both equity and benchmark
prices use simple `pct_change(fill_method=None)` returns computed before date
alignment, with no filling. All strategies use the same common, complete dates.
Inputs should have the same observation frequency and comparable return periods.
The first equity point supplies the starting level, so its return is excluded.

`--risk-free-rate` is an annual effective rate (default 0), converted to a periodic
rate as `(1 + rate) ** (1 / annualization) - 1`. OLS regresses strategy excess
returns on benchmark excess returns. Alpha is the intercept times
`--annualization` (default 252); active return is the mean return difference times
annualization. Tracking error and residual volatility use sample standard
deviations times `sqrt(annualization)`. Information ratio is active return divided
by tracking error, or NaN when periodic active-return standard deviation is at
most `1e-12`. R² is `1 - SSE / SST`; constant strategies have NaN correlation/R².
Summary analysis requires at least two common returns and raises an error for
benchmark variance at most `1e-16`.

Additional outputs are `factor_exposure.csv`, `rolling_beta.csv`, and
`rolling_correlation.csv`, plus `figures/strategy_vs_benchmark.png`,
`figures/rolling_beta.png`, and `figures/rolling_correlation.png`. The performance
plot normalizes original equity and benchmark levels to 1 on the first common
date. Rolling estimates require a full `--factor-window` of common observations
(default 60, minimum 2); incomplete or constant-benchmark windows are NaN.
These options affect factor reporting only, leaving existing strategy metrics intact.

For a local smoke test, use `data/raw/prices_demo.csv` for both CSVs with
`--benchmark-column AAPL --lookback 3 --rebalance 1 --clusters 2 --cap 0.9
--factor-window 3`. AAPL is also investable in that tiny fixture; this is only an
export smoke test, not a research experiment.

---

## Future Development

Planned improvements include:

* Graph neural network based portfolio signals
* Dynamic asset clustering
* Reinforcement learning portfolio allocation
* Interactive visualization dashboard
* Real-time market data integration

---

## License

MIT License
