import polars as pl
import numpy as np
from utils import save_lineplot, save_stackplot
from pathlib import Path
import datetime as dt

signal_names = ['reversal', 'momentum', 'bab']
start = dt.date(2001, 1, 1)
end = dt.date(2024, 12, 31)
WINDOW = 252
K = 3

# Prior hyperparameters
mu_0 = np.ones(K) / K
Sigma_0 = (1.0 / WINDOW) * np.eye(K)

# Load returns (mve)
returns_list = []
for signal_name in signal_names:
    returns = pl.read_parquet(f"data/mve_returns/{signal_name}.parquet")
    returns = returns.filter(pl.col('date').is_between(start, end)).sort('date')
    returns = returns.with_columns(pl.lit(signal_name).alias('name'))
    returns_list.append(returns)

returns: pl.DataFrame = pl.concat(returns_list)
returns = (
    returns
    .pivot(index='date', on='name', values='return')
    .sort('date')
)

# Compute signal weights using multivariate normal-normal conjugate prior on a rolling window
dates = returns['date'].to_list()
returns_np = returns.select(signal_names).to_numpy()

rows = []
for t in range(WINDOW, len(returns_np)):
    X = returns_np[t - WINDOW : t]
    n = X.shape[0]
    x_bar = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)

    # Posterior parameters (multivariate normal conjugate update)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_n = np.linalg.inv(Sigma_0_inv + n * Sigma_inv)
    mu_n = Sigma_n @ (Sigma_0_inv @ mu_0 + n * Sigma_inv @ x_bar)

    # Normalize
    weights = mu_n / mu_n.sum()

    # Softmax
    exp_w = np.exp(weights)
    weights = exp_w / exp_w.sum()

    rows.append(
        {'date': dates[t]}
        |
        {f'w_{name}': weights[i] for i, name in enumerate(signal_names)}
    )

# Combine and smooth weights
signal_weights = pl.DataFrame(rows)
signal_weights = (
    signal_weights
    .sort('date')
    .with_columns(
        pl.exclude('date').ewm_mean(span=WINDOW)
    )
)

# Save results
folder_path = Path("results/bayesian")
folder_path.mkdir(exist_ok=True, parents=True)

save_stackplot(
    signal_weights['date'],
    signal_weights['w_reversal'],
    signal_weights['w_momentum'],
    signal_weights['w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Bayesian",
)

save_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Bayesian",
)
