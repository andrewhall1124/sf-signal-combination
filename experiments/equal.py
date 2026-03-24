import polars as pl
import numpy as np
from utils import save_weights_lineplot, save_weights_stackplot, save_values_lineplot, save_returns_plot
from pathlib import Path
import datetime as dt

signal_names = ['reversal', 'momentum', 'bab']
start = dt.date(2001, 1, 1)
end = dt.date(2024, 12, 31)
plot_start = dt.date(2006, 1, 9)
WINDOW = 252 * 5

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

dates = returns.select('date').unique().sort('date')

signal_weights = (
    dates
    .with_columns(
        [
        pl.Series(f'w_{name}', [1.0 / len(signal_names)] * len(dates))
        for name in signal_names
        ]
    )
    .filter(
        pl.col('date').ge(plot_start)
    )
    .sort('date')
)

# Combine and smooth weights
weight_columns = [f'w_{name}' for name in signal_names]

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
folder_path = Path("results/equal")
folder_path.mkdir(exist_ok=True, parents=True)

save_returns_plot(
    returns=portfolio_returns,
    file_path=folder_path / "returns.png",
    title='Equal'
)

save_weights_stackplot(
    signal_weights['date'],
    signal_weights['w_reversal'],
    signal_weights['w_momentum'],
    signal_weights['w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Equal",
)

save_weights_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Equal",
)

