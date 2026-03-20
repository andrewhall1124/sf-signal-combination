import polars as pl
import numpy as np
from utils import save_weights_lineplot, save_weights_stackplot, save_values_lineplot, save_returns_plot
from pathlib import Path
import datetime as dt

signal_names = ['reversal', 'momentum', 'bab']
start = dt.date(2001, 1, 1)
end = dt.date(2024, 12, 31)
WINDOW = 252
K = 3

# Prior hyperparameters
mu_0 = np.ones(K) / K
n_0 = 1 # Pseudo observations
daily_variance = 1 / 100
Sigma_0 = (1.0 / n_0) * np.eye(K) * daily_variance

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

    print(Sigma - Sigma_0)

    weights_raw = np.linalg.solve(Sigma_n, mu_n)

    # Normalize
    weights_normalized = weights_raw / weights_raw.sum()

    # Softmax
    exp_w = np.exp(weights_normalized - weights_normalized.max())
    weights_softmax = exp_w / exp_w.sum()

    rows.append(
        {'date': dates[t]} |
        {f"w_raw_{name}": mu_n[i] for i, name in enumerate(signal_names)} |
        {f"w_norm_{name}": weights_normalized[i] for i, name in enumerate(signal_names)} |
        {f'w_{name}': weights_softmax[i] for i, name in enumerate(signal_names)}
    )

# Combine and smooth weights
weight_columns = [f'w_{name}' for name in signal_names]
signal_weights = pl.DataFrame(rows)
signal_weights = (
    signal_weights
    .sort('date')
    .with_columns(
        pl.col(weight_columns).ewm_mean(span=WINDOW)
    )
)

returns_long = (
    returns
    .unpivot(index='date', variable_name='signal_name', value_name='return')
    .sort('date', 'signal_name')
    .with_columns(
        pl.col('return').shift(-1).over('signal_name')
    )
)

signal_weights_long = (
    signal_weights
    .unpivot(index='date', on=weight_columns, variable_name='signal_name', value_name='weight')
    .with_columns(
        pl.col('signal_name').str.split('_').list.get(1)
    )
)

portfolio_returns = (
    signal_weights_long
    .join(
        other=returns_long,
        on=['date', 'signal_name'],
        how='left'
    )
    .drop_nulls('return')
    .group_by('date')
    .agg(
        pl.col('return').mul(pl.col('weight')).sum()
    )
    .sort('date')
)

# Save results
folder_path = Path("results/bayesian_nn")
folder_path.mkdir(exist_ok=True, parents=True)

save_returns_plot(
    returns=portfolio_returns,
    file_path=folder_path / "returns.png",
    title='Bayesian Normal-Normal'
)

save_values_lineplot(
    values=signal_weights,
    columns=['w_raw_reversal', 'w_raw_momentum', 'w_raw_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "values.png",
    title="Bayesian Normal-Normal",
    value_name="Raw Weights"
)

save_values_lineplot(
    values=signal_weights,
    columns=['w_norm_reversal', 'w_norm_momentum', 'w_norm_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "normalized.png",
    title="Bayesian Normal-Normal",
    value_name="Normalized Weights"
)

save_weights_stackplot(
    signal_weights['date'],
    signal_weights['w_reversal'],
    signal_weights['w_momentum'],
    signal_weights['w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Bayesian Normal-Normal",
)

save_weights_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Bayesian Normal-Normal",
)
