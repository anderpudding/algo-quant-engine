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