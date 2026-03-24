# sf-signal-combination

Research toolkit for comparing methods of allocating capital across multiple equity trading signals. Tests five weighting methodologies on three core signals to determine which combination approach yields the best risk-adjusted returns.

## Signals

Three trading signals are defined in `signals/expr.py`:

- **Momentum** — 230-day rolling sum of log returns, lagged 21 days. Captures trend persistence.
- **Reversal** — Negated 21-day rolling sum of log returns. Captures mean reversion.
- **BAB (Betting Against Beta)** — Inverse of predicted beta. Positions against systematic risk.

## Combination Methods

Each experiment in `experiments/` implements a different weighting strategy:

| Method | File | Description |
|--------|------|-------------|
| Equal | `experiments/equal.py` | Fixed 1/3 weight per signal (baseline) |
| MVE | `experiments/mve.py` | Mean-variance optimization on a rolling 5-year window |
| Bayesian NN | `experiments/bayesian_nn.py` | Normal-Normal conjugate prior with equal-weight prior |
| Bayesian NIW | `experiments/bayesian_niw.py` | Normal-Inverse-Wishart prior capturing covariance uncertainty |
| Fama-MacBeth | `experiments/fama_macbeth.py` | Two-stage cross-sectional regression |

Results (cumulative returns, weight time series, stacked weight charts) are saved to `results/{method}/`.

## Project Structure

```
signals/          Signal definitions, expression language, and return calculations
pipelines/        Data preparation — quantile returns, MVE backtests, portfolio weights
experiments/      Weighting method implementations and comparison scripts
results/          Generated plots and visualizations
```

## Setup

Requires Python 3.13. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### Dependencies

- `sf-quant` — Market data access
- `sf-backtester` — Portfolio optimization and backtesting
- `polars` / `polars-ols` — Dataframe operations and regression
- `matplotlib` / `seaborn` — Visualization

## Usage

### 1. Prepare data

Run pipelines to generate signal returns:

```bash
python pipelines/quantile_returns.py
python pipelines/mve_weights.py
python pipelines/mve_returns.py
```

### 2. Run experiments

```bash
python experiments/equal.py
python experiments/mve.py
python experiments/bayesian_nn.py
python experiments/bayesian_niw.py
python experiments/fama_macbeth.py
```

Each experiment writes plots to its corresponding `results/` subdirectory.
