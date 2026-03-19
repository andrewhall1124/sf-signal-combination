import polars as pl
import sf_quant.data as sfd
import datetime as dt
from signals.expr import momentum, reversal, bab
from signals.utils import get_assets_data

def compute_quantile_returns(signal_expr: pl.Expr, start: dt.date, end: dt.date) -> pl.DataFrame:
    assets = get_assets_data(start, end)
    
    signals = (
        assets
        .sort('date', 'barrid')
        .with_columns(
            signal=signal_expr
        )
    )

    filtered = (
        signals
        .sort('date', 'barrid')
        .filter(
            pl.col('signal').is_not_null(),
            pl.col('price').shift(1).over('barrid').gt(5)
        )
        .sort('date', 'barrid')
    )

    n_bins = 10
    top_bin = str(n_bins - 1)
    labels = [str(i) for i in range(n_bins)] 
    bins = (
        filtered
        .with_columns(
            pl.col('signal')
            .qcut(quantiles=10, labels=labels)
            .alias('bin')
        )
    )

    returns = (
        bins
        .group_by('date', 'bin')
        .agg(
            pl.col('specific_return').mean()
        )
        .sort('date', 'bin')
        .pivot(index='date', on='bin', values='specific_return')
        .with_columns(
            pl.col(top_bin).sub(pl.col('0')).alias('spread')
        )
        .select('date', 'spread')
        .rename({'spread': 'return'})
        .with_columns(
            pl.col('return').truediv(pl.col('return').std()) * .05 / pl.lit(252).sqrt()
        )
    )

    return returns
