import polars as pl
import datetime as dt
import polars_ols as pls
from signals.expr import momentum, reversal, bab
from signals.utils import get_assets_data
from experiments.utils import save_lineplot, save_stackplot
from pathlib import Path

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
signal_weights = (
    gammas
    .with_columns(
        w_momentum=pl.col('g_momentum') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
        w_reversal=pl.col('g_reversal') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
        w_bab=pl.col('g_bab') / pl.sum_horizontal('g_momentum', 'g_reversal', 'g_bab'),
    )
    .with_columns(
        e_mom=pl.col('w_momentum').exp(),
        e_rev=pl.col('w_reversal').exp(),
        e_bab=pl.col('w_bab').exp(),
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

# Save results
folder_path = Path("results/fama_macbeth")
folder_path.mkdir(exist_ok=True, parents=True)

save_stackplot(
    dates,
    w_reversal,
    w_momentum,
    w_bab,
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "stackplot.png",
    title="Fama Macbeth"
)

save_lineplot(
    weights=signal_weights,
    columns=['w_reversal', 'w_momentum', 'w_bab'],
    labels=['Reversal', 'Momentum', 'BAB'],
    file_path=folder_path / "lineplot.png",
    title="Fama Macbeth"
)
