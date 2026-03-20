import polars as pl
import numpy as np
from utils import save_weights_lineplot, save_weights_stackplot, save_values_lineplot
from pathlib import Path
import datetime as dt

signal_names = ['reversal', 'momentum', 'bab']
start = dt.date(2001, 1, 1)
end = dt.date(2024, 12, 31)
WINDOW = 252

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

    weights_raw = np.linalg.solve(Sigma, mu)

    # Normalize
    weights_normalized = weights_raw / weights_raw.sum()

    # Softmax (shift by max for numerical stability)
    exp_s = np.exp(weights_normalized - weights_normalized.max())
    weights_softmax = exp_s / exp_s.sum()

    rows.append(
        {'date': dates[t]} |
        {f'w_raw_{name}': weights_raw[i] for i, name in enumerate(signal_names)} |
        {f'w_norm_{name}': weights_normalized[i] for i, name in enumerate(signal_names)} |
        {f'w_{name}': weights_softmax[i] for i, name in enumerate(signal_names)}
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
folder_path = Path("results/mve")
folder_path.mkdir(exist_ok=True, parents=True)

save_values_lineplot(
    values=signal_weights,
    columns=['w_raw_reversal', 'w_raw_momentum', 'w_raw_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "values.png",
    title="MVE",
    value_name="Raw Weights"
)

save_values_lineplot(
    values=signal_weights,
    columns=['w_norm_reversal', 'w_norm_momentum', 'w_norm_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "normalized.png",
    title="MVE",
    value_name="Normalized Weights"
)

save_weights_stackplot(
    signal_weights['date'],
    signal_weights['w_reversal'],
    signal_weights['w_momentum'],
    signal_weights['w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="MVE",
)

save_weights_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="MVE",
)

