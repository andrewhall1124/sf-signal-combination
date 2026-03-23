import polars as pl
import datetime as dt
import polars_ols as pls
from signals.expr import momentum, reversal, bab
from signals.utils import get_assets_data
from experiments.utils import save_weights_lineplot, save_weights_stackplot, save_values_lineplot, save_returns_plot
from pathlib import Path

signal_names = ['reversal', 'momentum', 'bab']
start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
WINDOW = 252

assets = get_assets_data(start, end)

signals = (
    assets
    .with_columns(
        momentum=momentum(),
        reversal=reversal(),
        bab=bab()
    )
    .drop_nulls()
)

scores = (
    signals
    .unpivot(
        index=['date', 'barrid', 'price', 'return', 'specific_risk', 'forward_return'],
        variable_name='name',
        value_name='value'
    )
    .with_columns(
        score=pl.col('value')
        .sub(pl.col('value').mean())
        .truediv(pl.col('value').std())
        .over('date', 'name')
    )
)

alphas = (
    scores
    .with_columns(
        alpha=pl.lit(.05) * pl.col('specific_risk') * pl.col('score')
    )
)

regression_data = (
    alphas
    .pivot(
        index=['date', 'barrid', 'forward_return'],
        on='name',
        values='alpha'
    )
)

betas = (
    regression_data
    .sort('date', 'barrid')
    .with_columns(
        pl.lit(1.0).alias('const')
    )
    .with_columns(
        pl.col('forward_return')
        .least_squares
        .rolling_ols(
            pl.col('const', 'momentum', 'reversal', 'bab'),
            window_size=WINDOW,
            mode='coefficients'
        )
        .over('barrid')
        .alias('B')
    )
    .unnest('B', separator='_')
    .drop_nulls()
    .sort('date', 'barrid')
)

gammas = (
    betas
    .group_by('date')
    .agg(
        pl.col('forward_return')
        .least_squares
        .ols(pl.col('const', 'B_momentum', 'B_reversal', 'B_bab'), mode='coefficients')
        .alias('coefficients')
    )
    .unnest('coefficients')
    .rename({
        'const': 'g_const',
        'B_momentum': 'g_momentum',
        'B_reversal': 'g_reversal',
        'B_bab': 'g_bab'
    })
)

span = WINDOW
weight_columns = [f'w_{name}' for name in signal_names]
signal_weights = (
    gammas
    .with_columns(
        g_momentum=pl.col('g_momentum').abs(),
        g_reversal=pl.col('g_reversal').abs(),
        g_bab=pl.col('g_bab').abs(),
    )
    .with_columns(
        g_norm_momentum=pl.col('g_momentum') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
        g_norm_reversal=pl.col('g_reversal') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
        g_norm_bab=pl.col('g_bab') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
    )
    .with_columns(
        e_mom=pl.col('g_norm_momentum').exp(),
        e_rev=pl.col('g_norm_reversal').exp(),
        e_bab=pl.col('g_norm_bab').exp(),
    )
    .with_columns(
        w_momentum=pl.col('e_mom') / pl.sum_horizontal('e_mom', 'e_rev', 'e_bab'),
        w_reversal=pl.col('e_rev') / pl.sum_horizontal('e_mom', 'e_rev', 'e_bab'),
        w_bab=pl.col('e_bab') / pl.sum_horizontal('e_mom', 'e_rev', 'e_bab'),
    )
    .drop('e_mom', 'e_rev', 'e_bab')
    .with_columns(
        w_momentum=pl.col('w_momentum').ewm_mean(span=span),
        w_reversal=pl.col('w_reversal').ewm_mean(span=span),
        w_bab=pl.col('w_bab').ewm_mean(span=span)
    )
    .sort('date')
)

dates = signal_weights['date'].to_list()
w_reversal = signal_weights['w_reversal'].to_list()
w_momentum = signal_weights['w_momentum'].to_list()
w_bab = signal_weights['w_bab'].to_list()

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

returns_long = (
    returns
    .unpivot(index='date', variable_name='signal_name', value_name='return')
    .sort('date', 'signal_name')
    .with_columns(
        pl.col('return').shift(-2).over('signal_name')
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
folder_path = Path("results/fama_macbeth")
folder_path.mkdir(exist_ok=True, parents=True)

save_returns_plot(
    returns=portfolio_returns,
    file_path=folder_path / "returns.png",
    title='Fama-Macbeth'
)

save_values_lineplot(
    values=signal_weights,
    columns=['g_reversal', 'g_momentum', 'g_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "values.png",
    title="Fama-Macbeth",
    value_name="gamma"
)

save_values_lineplot(
    values=signal_weights,
    columns=['g_norm_reversal', 'g_norm_momentum', 'g_norm_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "normalized.png",
    title="Fama-Macbeth",
    value_name="Normalized gamma"
)

save_weights_stackplot(
    dates,
    w_reversal,
    w_momentum,
    w_bab,
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Fama Macbeth"
)

save_weights_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Fama Macbeth"
)
