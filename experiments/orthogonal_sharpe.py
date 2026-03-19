import polars as pl
import numpy as np
from utils import save_lineplot, save_stackplot
from pathlib import Path

signal_names = ['reversal', 'momentum', 'bab']
WINDOW = 252

# Load returns (quantile for now)
returns_list = []
for signal_name in signal_names:
    returns = pl.read_parquet(f"data/quantile_returns/{signal_name}.parquet")
    returns = returns.with_columns(pl.lit(signal_name).alias('name'))
    returns_list.append(returns)

returns: pl.DataFrame = pl.concat(returns_list)
returns = (
    returns
    .pivot(index='date', on='name', values='return')
    .drop_nulls()
    .sort('date')
)

# Compute signal weights using analytical MVE on a rolling window
dates = returns['date'].to_list()
R_full = returns.select(signal_names).to_numpy()

rows = []
for t in range(WINDOW, len(R_full)):
    R_window = R_full[t - WINDOW : t]

    mu = R_window.mean(axis=0)
    Sigma = np.cov(R_window, rowvar=False)

    weights = np.linalg.solve(Sigma, mu)

    # Normalize
    weights = weights / weights.sum()

    # Softmax
    exp_s = np.exp(weights)
    weights = exp_s / exp_s.sum()

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
folder_path = Path("results/orthogonal_sharpe")
folder_path.mkdir(exist_ok=True, parents=True)

save_stackplot(
    signal_weights['date'],
    signal_weights['w_reversal'],
    signal_weights['w_momentum'],
    signal_weights['w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Orthogonal Sharpe",
)

save_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Orthogonal Sharpe",
)

